"""Load inputs and validate generated calls without loading an LLM."""

import json
import logging
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from src.models.function_def import FunctionDef
from src.models.prompt import FunctionCallResult, PromptItem


def read_json_file(file_path: str | Path) -> object:
    """Read JSON without assuming its schema.

    Args:
        file_path: Path to a UTF-8 JSON file.

    Returns:
        The decoded JSON value, ready for model validation.

    Raises:
        ValueError: If the file cannot be read or decoded.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data: object = json.load(file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read JSON file '{file_path}': {error}"
        ) from error
    return data


def load_function_definitions(
    file_path: str | Path,
) -> dict[str, FunctionDef]:
    """Validate all definitions before returning a registry keyed by name.

    Args:
        file_path: Path to a JSON array of function definitions.

    Returns:
        Validated definitions indexed by their unique function names.

    Raises:
        ValueError: If reading, validation, or duplicate-name checks fail.
    """
    data = read_json_file(file_path)
    try:
        definitions = TypeAdapter(list[FunctionDef]).validate_python(data)
    except ValidationError as error:
        raise ValueError(
            f"Invalid function definitions in '{file_path}': {error}"
        ) from error

    functions: dict[str, FunctionDef] = {}
    seen: set[str] = set()
    for definition in definitions:
        if definition.name in seen:
            raise ValueError(
                f"Duplicate function '{definition.name}' in '{file_path}'."
            )
        seen.add(definition.name)
        unsupported = {
            param.type for param in definition.parameters.values()
            if param.type not in {
                "string", "number", "integer", "boolean", "null",
            }
        }
        if unsupported:
            logging.getLogger(__name__).warning(
                "Function '%s' excluded: unsupported parameter types: %s",
                definition.name, ", ".join(sorted(unsupported)),
            )
            continue
        functions[definition.name] = definition
    return functions


def load_prompt_items(file_path: str | Path) -> list[PromptItem]:
    """Validate a prompt array, preserving its order and repeated entries.

    Args:
        file_path: Path to a JSON array of prompt objects.

    Returns:
        Validated prompts in input order.

    Raises:
        ValueError: If the file cannot be read or its prompts are invalid.
    """
    data = read_json_file(file_path)
    try:
        return TypeAdapter(list[PromptItem]).validate_python(data)
    except ValidationError as error:
        raise ValueError(
            f"Invalid prompts in '{file_path}': {error}"
        ) from error


def validate_function_calls(
    data: object,
    functions: dict[str, FunctionDef],
    prompts: list[PromptItem],
) -> list[FunctionCallResult]:
    """Validate generated calls against definitions and original prompts.

    Args:
        data: Decoded JSON array produced by the generation pipeline.
        functions: Registry returned by load_function_definitions.
        prompts: Original prompts in input order.

    Returns:
        One validated call for each original prompt.

    Raises:
        ValidationError: If an output violates its schema or contract.
        ValueError: If call counts or original prompt texts do not match.
    """

    calls = TypeAdapter(list[FunctionCallResult]).validate_python(
        data, context={"functions": functions}
    )
    if len(calls) != len(prompts):
        raise ValueError("Expected exactly one function call per prompt.")
    for index, (call, prompt) in enumerate(zip(calls, prompts)):
        if call.prompt != prompt.prompt:
            raise ValueError(
                f"Original prompt does not match at index {index}."
            )

    return calls
