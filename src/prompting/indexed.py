"""Build indexed-label contexts for function-selection experiments."""

from collections.abc import Collection

from src.models.function_def import FunctionDef
from src.prompting.sanitize import safe_json

_SYSTEM = (
    "Choose exactly one function for the request. Reply only with its "
    "letter label."
)
_PREFILL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def indexed_context(
    functions: list[FunctionDef],
    request: str,
    markers: Collection[str] = (),
) -> tuple[str, dict[str, str]]:
    """Build a letter-indexed function list and its label mapping.

    Args:
        functions: Function definitions in the desired registry order.
        request: Original user request.
        markers: External token spellings that must be escaped in JSON.

    Returns:
        The complete Qwen context and a mapping from labels to function names.

    Raises:
        ValueError: If the function list is empty or exceeds 26 entries.
    """
    if not functions:
        raise ValueError("At least one function is required.")
    if len(functions) > 26:
        raise ValueError("Indexed selection supports at most 26 functions.")

    safe_markers = tuple(markers)
    labels = {
        chr(ord("A") + index): function.name
        for index, function in enumerate(functions)
    }
    descriptions: list[str] = []
    for index, function in enumerate(functions):
        label = chr(ord("A") + index)
        parameters = {
            key: {
                "type": parameter.type,
                **({"description": parameter.description}
                   if parameter.description is not None else {}),
            }
            for key, parameter in function.parameters.items()
        }
        description = {
            "name": function.name,
            "description": function.description,
            "parameters": parameters,
        }
        descriptions.append(
            label + " " + safe_json(description, safe_markers)
        )
    user_content = "\n".join(descriptions)
    user_content += "\nRequest: " + safe_json(request, safe_markers)
    context = (
        "<|im_start|>system\n" + _SYSTEM + "<|im_end|>\n"
        "<|im_start|>user\n" + user_content + "<|im_end|>\n" + _PREFILL
    )
    return context, labels
