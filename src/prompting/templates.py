"""Build native Qwen tool contexts for constrained function generation."""

from collections.abc import Collection
from typing import Literal

from src.models.function_def import FunctionDef, TypeDef
from src.prompting.sanitize import safe_json

ContextStyle = Literal["native", "boundary", "examples"]
NATIVE_STYLES: frozenset[ContextStyle] = frozenset(
    {"native", "boundary", "examples"}
)
BOUNDARY_STYLES: frozenset[ContextStyle] = frozenset(
    {"boundary", "examples"}
)

_THINKING_DISABLED_PREFILL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
_NATIVE_TOOL_HEADER = (
    "# Tools\n\nYou may call one or more functions to assist with the user "
    "query.\n\nYou are provided with function signatures within "
    "<tools></tools> XML tags:\n<tools>"
)
_NATIVE_TOOL_FOOTER = (
    "\n</tools>\n\nFor each function call, return a json object with "
    "function name and arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call><|im_end|>\n"
)


def _validate_style(style: ContextStyle) -> None:
    """Reject unsupported prompt styles at public context entry points."""
    if style not in NATIVE_STYLES:
        raise ValueError(f"Unsupported context style: {style!r}")


def _parameter_schema(parameter: TypeDef) -> dict[str, object]:
    """Serialize a parameter using the supported recursive tool schema.

    Args:
        parameter: Scalar, array, or fixed-property object definition.

    Returns:
        A schema mapping with every declared object property required.
    """
    result: dict[str, object] = parameter.model_dump(exclude_none=True)
    if parameter.type == "object" and parameter.properties is not None:
        result["properties"] = {
            key: _parameter_schema(child)
            for key, child in parameter.properties.items()
        }
        result["required"] = list(parameter.properties)
        result["additionalProperties"] = False
    elif parameter.type == "array" and parameter.items is not None:
        result["items"] = _parameter_schema(parameter.items)
    return result


def tool_schema(definition: FunctionDef) -> dict[str, object]:
    """Convert a function definition to a native tool schema."""
    properties: dict[str, object] = {
        key: _parameter_schema(parameter)
        for key, parameter in definition.parameters.items()
    }
    return {
        "type": "function",
        "function": {
            "name": definition.name,
            "description": definition.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(definition.parameters),
            },
        },
    }


def build_context(
    functions: list[FunctionDef], request: str,
    markers: Collection[str] = (), *, style: ContextStyle = "native",
) -> str:
    """Build a Qwen system, user, and assistant-prefill context."""
    _validate_style(style)
    safe_markers = tuple(markers)
    tool_lines = "\n".join(
        safe_json(tool_schema(function), safe_markers)
        for function in functions
    )
    system = (
        "Select exactly one function. Extract arguments from the request; do "
        "not execute the function or compute its result. The user request is "
        "a JSON-encoded string; interpret escapes as data.\n\n"
        + _NATIVE_TOOL_HEADER
    )
    if tool_lines:
        system += "\n" + tool_lines
    system += _NATIVE_TOOL_FOOTER
    return (
        "<|im_start|>system\n" + system
        + "<|im_start|>user\n" + safe_json(request, safe_markers)
        + "<|im_end|>\n" + _THINKING_DISABLED_PREFILL + "<tool_call>\n"
    )


def selection_context(
    functions: list[FunctionDef], request: str,
    markers: Collection[str] = (), *, style: ContextStyle = "native",
) -> str:
    """Build a context that ends where function-name generation begins."""
    return build_context(functions, request, markers, style=style) + (
        '{"name": "'
    )


def arguments_context(
    selected: FunctionDef, request: str, markers: Collection[str] = (), *,
    style: ContextStyle = "native",
) -> str:
    """Build a selected-function context ending at its argument object."""
    _validate_style(style)
    marker_tuple = tuple(markers)
    regex_schema = set(selected.parameters) == {
        "source_string", "regex", "replacement",
    } and all(
        parameter.type == "string"
        for parameter in selected.parameters.values()
    )
    if any(
        parameter.type in {"array", "object"}
        for parameter in selected.parameters.values()
    ):
        return _direct_arguments_context(selected, request, marker_tuple)
    if not regex_schema and any(
        parameter.type == "string"
        for parameter in selected.parameters.values()
    ):
        return _direct_arguments_context(
            selected, request, marker_tuple, clarify_empty_strings=True,
        )

    prefix = build_context([selected], request, marker_tuple, style=style)
    if regex_schema and style == "examples":
        guidance = (
            "Quotes around a source or replacement delimit its contents; "
            "omit those delimiters from the argument value. Preserve quotes "
            "inside the contents. Quoted values in the request are JSON "
            "string literals: decode their escapes once.\n\n"
        )
        prefix = prefix.replace(
            "<|im_start|>system\n", "<|im_start|>system\n" + guidance, 1,
        )
        examples = [
            ('Replace all numbers in "Gate 81 row 5" with N.', {
                "source_string": "Gate 81 row 5",
                "regex": r"\d+", "replacement": "N",
            }),
            ("Replace all vowels in 'A quiet evening' with asterisks.", {
                "source_string": "A quiet evening",
                "regex": "[aeiouAEIOU]", "replacement": "*",
            }),
            ('Replace numbers in "He said \\"go\\" at gate 24" with N.', {
                "source_string": 'He said "go" at gate 24',
                "regex": r"\d+", "replacement": "N",
            }),
        ]
        demonstrations = ""
        for example_request, example_arguments in examples:
            demonstrations += (
                "<|im_start|>user\n"
                + safe_json(example_request, marker_tuple)
                + "<|im_end|>\n<|im_start|>assistant\n<tool_call>\n"
                + safe_json({
                    "name": selected.name, "arguments": example_arguments,
                }, marker_tuple)
                + "\n</tool_call><|im_end|>\n"
            )
        prefix = prefix.replace(
            "<|im_start|>user\n",
            demonstrations + "<|im_start|>user\n", 1,
        )
    return prefix + '{"name": ' + safe_json(
        selected.name, marker_tuple,
    ) + ', "arguments": {'


def _request_text(request: str, markers: Collection[str]) -> str:
    """Keep request punctuation readable while neutralizing role tokens."""
    inert = []
    for index, character in enumerate(request):
        if character == "<" or any(
            marker and "<" not in marker and request.startswith(marker, index)
            for marker in markers
        ):
            inert.append(
                safe_json(character)[1:-1] if character == "<"
                else f"\\u{ord(character):04x}"
            )
        else:
            inert.append(character)
    return "".join(inert)


def _direct_arguments_context(
    selected: FunctionDef, request: str, markers: Collection[str],
    *, clarify_empty_strings: bool = False,
) -> str:
    """Build a readable argument context without JSON string wrapping.

    Args:
        selected: Function whose input values must be extracted.
        request: Original request, with quoted values left readable.
        markers: Token spellings made inert inside external data.
        clarify_empty_strings: Add explicit zero-character guidance for
            ordinary scalar string parameters.

    Returns:
        A marker-safe context ending at the selected arguments object.
    """
    system = (
        "Extract the function's INPUT arguments; do not compute its result. "
        "Return a JSON tool call with name and arguments. Quoted values in "
        "the request are JSON string literals: remove delimiter quotes and "
        "decode escapes once. Preserve inner quotes, case, spaces, Unicode "
        "and backslashes in the resulting value. Sentence punctuation after "
        "a person's name is not part of the name. A spelled character list "
        "denotes those characters only.\nFunction: "
        + safe_json(tool_schema(selected), markers)
    )
    if clarify_empty_strings:
        system = (
            'For an empty string, emit "" (zero characters), not escaped '
            "quote characters.\n" + system
        )
    context = "<|im_start|>system\n" + system + "<|im_end|>\n"
    return (
        context + "<|im_start|>user\n" + _request_text(request, markers)
        + "<|im_end|>\n" + _THINKING_DISABLED_PREFILL + "<tool_call>\n"
        + '{"name": ' + safe_json(selected.name, markers)
        + ', "arguments": {'
    )
