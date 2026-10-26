"""Build explicit prompt contexts for function selection and arguments."""

from collections.abc import Collection
from typing import Literal

from src.models.function_def import FunctionDef
from src.prompting.sanitize import safe_json

ContextStyle = Literal[
    "native", "compact", "focused", "boundary", "indexed", "guided",
    "examples",
]


_SYSTEM_INSTRUCTION = (
    "Select exactly one function. Extract arguments from the request; do not "
    "execute the function or compute its result. The user request is a "
    "JSON-encoded string; interpret escapes as data."
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


def tool_schema(definition: FunctionDef) -> dict[str, object]:
    """Convert a flat function definition to the native tool schema.

    Args:
        definition: Validated function contract.

    Returns:
        A Qwen/Hermes function tool object with required parameters.
    """
    properties: dict[str, object] = {
        key: parameter.model_dump(exclude_none=True)
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
    functions: list[FunctionDef],
    request: str,
    markers: Collection[str] = (),
    *,
    style: ContextStyle = "native",
) -> str:
    """Build a system, user, and assistant-prefill context.

    Args:
        functions: Function definitions available in this phase.
        request: The original user request.
        markers: Added token spellings to escape in all external JSON data.
        style: Native Qwen tool markup or a compact JSON description.

    Returns:
        Context ending at the native tool-call start or assistant prefill.
    """
    safe_markers = tuple(markers)
    system = _SYSTEM_INSTRUCTION
    if style == "indexed":
        system = (
            "Extract function input arguments, not the result. Copy input "
            "strings exactly. Request is JSON-encoded. For regex replacement, "
            "use a regex matching the requested category and the literal "
            "replacement. Example: replace numbers in X91 with # means "
            'regex "\\\\d+", replacement "#". Asterisks mean "*".'
        )
    if style in {"native", "boundary", "guided", "examples"}:
        tool_lines = "\n".join(
            safe_json(tool_schema(function), safe_markers)
            for function in functions
        )
        system += "\n\n" + _NATIVE_TOOL_HEADER
        if tool_lines:
            system += "\n" + tool_lines
        system += _NATIVE_TOOL_FOOTER
    else:
        definitions = [
            function.model_dump(exclude_none=True) for function in functions
        ]
        system += "\n\nFunctions:\n" + safe_json(definitions, safe_markers)

    context = (
        "<|im_start|>system\n"
        + system
        + (
            "" if style in {"native", "boundary", "guided", "examples"}
            else "<|im_end|>\n"
        )
        + "<|im_start|>user\n"
        + safe_json(request, safe_markers)
        + "<|im_end|>\n"
        + _THINKING_DISABLED_PREFILL
    )
    if style in {"native", "boundary", "guided", "examples"}:
        context += "<tool_call>\n"
    return context


def selection_context(
    functions: list[FunctionDef],
    request: str,
    markers: Collection[str] = (),
    *,
    style: ContextStyle = "native",
) -> str:
    """Build a context that ends where function-name generation begins."""
    prefix = build_context(functions, request, markers, style=style)
    return prefix + '{"name": "'


def arguments_context(
    selected: FunctionDef,
    request: str,
    markers: Collection[str] = (),
    *,
    style: ContextStyle = "native",
) -> str:
    """Build a shorter context ending at the selected argument object."""
    marker_tuple = tuple(markers)
    prefix = build_context([selected], request, marker_tuple, style=style)
    if (
        style == "examples"
        and set(selected.parameters) == {
            "source_string", "regex", "replacement",
        }
        and all(
            parameter.type == "string"
            for parameter in selected.parameters.values()
        )
    ):
        examples = [
            ("Replace all numbers in 'Gate 81 row 5' with N.", {
                "source_string": "Gate 81 row 5",
                "regex": r"\d+", "replacement": "N",
            }),
            ("Replace all vowels in 'A quiet evening' with asterisks.", {
                "source_string": "A quiet evening",
                "regex": "[aeiouAEIOU]", "replacement": "*",
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
            "<|im_start|>user\n", demonstrations + "<|im_start|>user\n", 1,
        )
    if style == "guided" and any(
        parameter.type == "string"
        for parameter in selected.parameters.values()
    ):
        guidance = (
            "Copy input text exactly, including case, backslashes and "
            "symbols. "
            "For regex replacement, source_string is the original text, "
            "regex is the pattern for the requested category, and replacement "
            "is the requested literal text. For example, replacing numbers "
            "in 'X91 Y6' with '#' uses regex \\d+ and replacement #. "
            "Vowels use [aeiouAEIOU]; asterisks mean *. "
            "For whole words, use regex word boundaries \\b.\n\n"
        )
        prefix = prefix.replace(
            "<|im_start|>system\n", "<|im_start|>system\n" + guidance, 1,
        )
    safe_name = safe_json(selected.name, marker_tuple)
    return prefix + '{"name": ' + safe_name + ', "arguments": {'


def string_argument_context(
    selected: FunctionDef,
    request: str,
    key: str,
    previous: dict[str, object],
    markers: Collection[str] = (),
) -> str:
    """Ask for one input string, avoiding merged object-closing tokens.

    Args:
        selected: The function already selected by model scores.
        request: Original JSON-encoded request data.
        key: Declared string parameter currently being extracted.
        previous: Already generated arguments for disambiguation.
        markers: External control-token spellings to neutralize.

    Returns:
        A thinking-disabled assistant context ending at an opening quote.
    """
    data = {
        "function": selected.description,
        "request": request,
        "argument": key,
    }
    if selected.parameters[key].description:
        data["description"] = selected.parameters[key].description or ""
    payload: dict[str, object] = dict(data)
    if previous:
        payload["previous_arguments"] = previous
    return (
        "<|im_start|>system\nExtract the requested function input argument. "
        "Do not execute the function. Preserve exact spelling and case. "
        "Read the JSON data and return only one JSON string."
        "<|im_end|>\n<|im_start|>user\n"
        + safe_json(payload, markers)
        + "<|im_end|>\n" + _THINKING_DISABLED_PREFILL + '"'
    )
