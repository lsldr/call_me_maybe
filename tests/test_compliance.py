"""Enforce mandatory-source import and public-SDK access restrictions."""

import ast
from pathlib import Path

import pytest


FORBIDDEN_IMPORTS = {
    "dspy", "huggingface", "huggingface_hub", "outlines", "pytorch",
    "torch", "transformers", "torchvision", "tensorflow", "keras",
    "pytorch_lightning",
}


def compliance_violations(source: str) -> list[str]:
    """Find forbidden imports and private access on identifiable SDK objects.

    Track SDK imports, constructor assignments, annotations, and aliases.
    This guards ordinary source; it is not a dynamic Python sandbox.
    """
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    violations: list[str] = []
    sdk_objects: set[str] = set()
    constructors: set[str] = set()

    def key(node: ast.AST) -> str:
        return ast.dump(node, include_attributes=False)

    def report(node: ast.AST, message: str) -> None:
        violations.append(f"line {getattr(node, 'lineno', '?')}: {message}")

    for node in nodes:
        imports: list[str] = []
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
            for alias in node.names:
                if alias.name == "llm_sdk":
                    name = alias.asname or alias.name
                    sdk_objects.add(key(ast.Name(id=name, ctx=ast.Load())))
                    constructors.add(key(ast.Attribute(
                        value=ast.Name(id=name, ctx=ast.Load()),
                        attr="Small_LLM_Model", ctx=ast.Load(),
                    )))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports = [module]
            if module == "llm_sdk":
                for alias in node.names:
                    if alias.name == "Small_LLM_Model":
                        name = alias.asname or alias.name
                        constructors.add(key(ast.Name(
                            id=name, ctx=ast.Load(),
                        )))
                    if alias.name.startswith("_"):
                        report(node, f"private SDK import: {alias.name}")
        for module in imports:
            if module.split(".")[0] in FORBIDDEN_IMPORTS:
                report(node, f"forbidden import: {module}")
            if module.startswith("llm_sdk.") and any(
                part.startswith("_") for part in module.split(".")[1:]
            ):
                report(node, f"private SDK import: {module}")

    def is_sdk(node: ast.AST) -> bool:
        return key(node) in sdk_objects or (
            isinstance(node, ast.Call) and key(node.func) in constructors
        )

    # Iterate so aliases are resolved independently of AST traversal order.
    changed = True
    while changed:
        previous = len(sdk_objects)
        for node in nodes:
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign) and is_sdk(node.value):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and (
                key(node.annotation) in constructors
                or (node.value is not None and is_sdk(node.value))
            ):
                targets = [node.target]
            elif isinstance(node, ast.arg) and node.annotation is not None:
                if key(node.annotation) in constructors:
                    sdk_objects.add(key(ast.Name(
                        id=node.arg, ctx=ast.Load(),
                    )))
            for target in targets:
                if isinstance(target, (ast.Name, ast.Attribute)):
                    # Assignment targets use Store; accesses use Load.
                    target.ctx = ast.Load()
                    sdk_objects.add(key(target))
        changed = len(sdk_objects) != previous

    for node in nodes:
        if isinstance(node, ast.Attribute) and is_sdk(node.value):
            if node.attr.startswith("_"):
                report(node, f"private SDK attribute: {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"getattr", "setattr", "delattr", "hasattr"}:
                if len(node.args) >= 2 and is_sdk(node.args[0]):
                    attr = node.args[1]
                    if isinstance(attr, ast.Constant):
                        if (isinstance(attr.value, str)
                                and attr.value.startswith("_")):
                            report(node, "private SDK attribute: "
                                   f"{attr.value}")
    return violations


def test_mandatory_source_complies() -> None:
    """Scan all application files and fail with file/line diagnostics."""
    root = Path(__file__).resolve().parents[1] / "src"
    files = sorted(root.rglob("*.py"))
    assert files, "No application source files found"
    violations = [
        f"{path.relative_to(root)}: {violation}"
        for path in files
        for violation in compliance_violations(
            path.read_text(encoding="utf-8"))
    ]
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("source", [
    "import torch",
    "import numpy, transformers.models as models",
    "from huggingface_hub import download",
    "def lazy():\n    from outlines import generate",
    "from llm_sdk import _secret",
    "import llm_sdk._internal",
    "from llm_sdk import Small_LLM_Model as Model\nm = Model()\nm._model",
    "import llm_sdk as sdk\nsdk.Small_LLM_Model()._tokenizer",
    "from llm_sdk import Small_LLM_Model\n"
    "def probe(model: Small_LLM_Model):\n    return model._device",
    "from llm_sdk import Small_LLM_Model\n"
    "self._sdk = Small_LLM_Model()\ncopy = self._sdk\ncopy._anything()",
    "from llm_sdk import Small_LLM_Model\n"
    "model = Small_LLM_Model()\ngetattr(model, '_tokenizer')",
])
def test_detector_rejects_violations(source: str) -> None:
    """Prove violations fail detection, including aliased and nested use."""
    assert compliance_violations(source)


@pytest.mark.parametrize("source", [
    "import numpy\nfrom pathlib import Path",
    "import torch_tools",  # Match package names, not arbitrary prefixes.
    "from llm_sdk import Small_LLM_Model\n"
    "self._sdk = Small_LLM_Model()\nself._sdk.decode([1])",
    "self._cache = {}\nself._private_helper()",
    "# import torch\ntext = 'model._tokenizer'",
])
def test_detector_accepts_public_access(source: str) -> None:
    """Do not confuse local private attributes or text with SDK violations."""
    assert compliance_violations(source) == []
