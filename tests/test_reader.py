"""Exercise input loading and function contracts without loading an LLM."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.io.reader import (
    load_function_definitions,
    load_prompt_items,
    read_json_file,
    validate_function_calls,
)
from src.models.function_def import FunctionDef
from src.models.prompt import FunctionCallResult, PromptItem


@pytest.fixture
def functions() -> dict[str, FunctionDef]:
    """Load the project's real sample definitions."""
    root = Path(__file__).resolve().parents[1]
    return load_function_definitions(
        root / "data/input/functions_definition.json"
    )


def test_sample_call(functions: dict[str, FunctionDef]) -> None:
    """Validate realistic output against the supplied input files."""
    root = Path(__file__).resolve().parents[1]
    prompts = load_prompt_items(
        root / "data/input/function_calling_tests.json"
    )
    data = [{
        "prompt": prompts[0].prompt,
        "name": "fn_add_numbers",
        "parameters": {"a": 2, "b": 3.0},
    }]
    calls = validate_function_calls(data, functions, prompts[:1])
    assert calls[0].model_dump() == data[0]


@pytest.mark.parametrize(
    "value", ["3", True, None, float("nan"), float("inf")]
)
def test_invalid_numeric_argument(
    functions: dict[str, FunctionDef], value: object
) -> None:
    """Reject coercion and invalid numbers, including the second argument."""
    with pytest.raises(ValidationError):
        FunctionCallResult.model_validate(
            {
                "prompt": "Add numbers",
                "name": "fn_add_numbers",
                "parameters": {"a": 2, "b": value},
            },
            context={"functions": functions},
        )


@pytest.mark.parametrize("change", [
    {"name": "unknown"},
    {"parameters": {"a": 2}},
    {"parameters": {"a": 2, "b": 3, "extra": 4}},
    {"description": "Unexpected field"},
    {"parameters": []},
])
def test_invalid_call_contract(
    functions: dict[str, FunctionDef], change: dict[str, object]
) -> None:
    """Reject unknown functions, wrong keys, and malformed parameters."""
    data = {
        "prompt": "Add numbers",
        "name": "fn_add_numbers",
        "parameters": {"a": 2, "b": 3},
        **change,
    }
    with pytest.raises(ValidationError):
        FunctionCallResult.model_validate(
            data, context={"functions": functions}
        )


@pytest.mark.parametrize(("type_name", "value"), [
    ("string", ""), ("boolean", False), ("number", 0), ("number", -1.5),
])
def test_supported_scalar_types(type_name: str, value: object) -> None:
    """Accept each supported type, including empty strings and false."""
    definition = FunctionDef.model_validate({
        "name": "example",
        "description": "Example function",
        "parameters": {"value": {"type": type_name}},
        "returns": {"type": type_name},
    })
    call = FunctionCallResult.model_validate(
        {"prompt": "", "name": "example", "parameters": {"value": value}},
        context={"functions": {"example": definition}},
    )
    assert call.parameters["value"] == value


def test_zero_parameters_and_required_context() -> None:
    """Allow no-argument functions while requiring a registry for calls."""
    definition = FunctionDef.model_validate({
        "name": "example", "description": "No arguments",
        "parameters": {}, "returns": {"type": "string"},
    })
    data = {"prompt": "", "name": "example", "parameters": {}}
    with pytest.raises(ValidationError, match="context"):
        FunctionCallResult.model_validate(data)
    call = FunctionCallResult.model_validate(
        data, context={"functions": {"example": definition}}
    )
    assert call.parameters == {}


def test_input_arrays_and_repeated_prompts(tmp_path: Path) -> None:
    """Keep all prompts in order and reject a top-level object."""
    path = tmp_path / "prompts.json"
    data = [{"prompt": "hello"}, {"prompt": ""}, {"prompt": "hello"}]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert [item.model_dump() for item in load_prompt_items(path)] == data
    path.write_text('{"prompt": "hello"}', encoding="utf-8")
    with pytest.raises(ValueError, match="prompts.json"):
        load_prompt_items(path)
    with pytest.raises(ValueError, match="prompts.json"):
        load_function_definitions(path)


def test_duplicate_and_invalid_definitions(
    tmp_path: Path, functions: dict[str, FunctionDef]
) -> None:
    """Reject duplicate names and files containing invalid entries."""
    path = tmp_path / "functions.json"
    definition = functions["fn_greet"].model_dump()
    path.write_text(json.dumps([definition, definition]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate function"):
        load_function_definitions(path)
    path.write_text(json.dumps([definition, {}]), encoding="utf-8")
    with pytest.raises(ValueError, match="functions.json"):
        load_function_definitions(path)


def test_file_errors(tmp_path: Path) -> None:
    """Include the offending path in file and decoding errors."""
    path = tmp_path / "input.json"
    with pytest.raises(ValueError, match="input.json"):
        read_json_file(path)
    for content in (b"[", b"\xff"):
        path.write_bytes(content)
        with pytest.raises(ValueError, match="input.json"):
            read_json_file(path)


def test_output_correspondence(functions: dict[str, FunctionDef]) -> None:
    """Require exactly one result per prompt with unchanged prompt text."""
    prompts = [PromptItem(prompt="hello")]
    with pytest.raises(ValueError, match="one function call"):
        validate_function_calls([], functions, prompts)
    with pytest.raises(ValueError, match="Original prompt"):
        validate_function_calls(
            [{"prompt": "changed", "name": "fn_greet",
              "parameters": {"name": "Shrek"}}],
            functions,
            prompts,
        )


@pytest.mark.parametrize(("type_name", "value"), [
    ("integer", 2), ("integer", -3), ("null", None),
])
def test_additional_scalar_types(type_name: str, value: object) -> None:
    """Accept integer and null contracts without coercion."""
    test_supported_scalar_types(type_name, value)


@pytest.mark.parametrize(("type_name", "value"), [
    ("integer", True), ("integer", 2.0), ("integer", "2"),
    ("null", False), ("null", 0), ("null", "null"),
])
def test_additional_scalar_rejections(type_name: str, value: object) -> None:
    """Reject values of the wrong exact scalar type."""
    with pytest.raises(ValidationError):
        test_supported_scalar_types(type_name, value)


def test_recursive_and_unsupported_schemas(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Preserve nested schemas and warn about unavailable functions."""
    raw = {
        "name": "nested", "description": "Nested input",
        "parameters": {"value": {
            "type": "array", "items": {
                "type": "object", "properties": {"x": {"type": "integer"}},
            }, "minItems": 1,
        }},
        "returns": {"type": "object"},
    }
    definition = FunctionDef.model_validate(raw)
    schema = definition.parameters["value"]
    assert schema.items is not None
    assert schema.items.properties is not None
    assert schema.items.properties["x"].type == "integer"
    assert schema.model_extra == {"minItems": 1}
    supported = {
        **raw, "name": "supported", "parameters": {},
    }
    path = tmp_path / "functions.json"
    path.write_text(json.dumps([raw, supported]), encoding="utf-8")
    assert set(load_function_definitions(path)) == {"supported"}
    assert "nested" in caplog.text
    assert "unsupported parameter types: array" in caplog.text
    path.write_text(json.dumps([raw]), encoding="utf-8")
    assert load_function_definitions(path) == {}
