"""Integration checks for the optional nested-schema profile."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import RunSettings
from src.io.reader import load_function_definitions, validate_function_calls
from src.models.prompt import PromptItem
from src.schemas.nested import SchemaLimits


def _write_definitions(tmp_path: Path, definitions: object) -> Path:
    path = tmp_path / "functions.json"
    path.write_text(json.dumps(definitions), encoding="utf-8")
    return path


def _function(parameters: dict[str, object]) -> dict[str, object]:
    return {
        "name": "submit",
        "description": "Submit nested data",
        "parameters": parameters,
        "returns": {"type": "null"},
    }


def _nested_definition() -> dict[str, object]:
    return _function({
        "payload": {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "score": {"type": "number"},
                        },
                        "required": ["label", "score"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                },
            },
            "required": ["records"],
            "additionalProperties": False,
        },
    })


def test_mandatory_loader_excludes_container_parameters(
    tmp_path: Path,
) -> None:
    path = _write_definitions(tmp_path, [
        _function({"text": {"type": "string"}}),
        {
            **_function({"items": {
                "type": "array", "items": {"type": "string"},
            }}),
            "name": "array_fn",
        },
        {
            **_function({"details": {
                "type": "object", "properties": {"x": {"type": "integer"}},
            }}),
            "name": "nested",
        },
    ])

    functions = load_function_definitions(path)

    assert list(functions) == ["submit"]


def test_bonus_loader_accepts_valid_recursive_schema(tmp_path: Path) -> None:
    path = _write_definitions(tmp_path, [_nested_definition()])

    functions = load_function_definitions(
        path, allow_nested=True, schema_limits=SchemaLimits(),
    )

    assert list(functions) == ["submit"]
    assert functions["submit"].parameters["payload"].properties is not None


@pytest.mark.parametrize("parameter", [
    {"type": "object", "properties": {"x": {"type": "string"}},
     "required": []},
    {"type": "string", "anyOf": [{"type": "string"}, {"type": "null"}]},
    {"type": "string", "$ref": "#/definitions/Text"},
])
def test_bonus_loader_rejects_unsupported_schema_before_model(
    tmp_path: Path, parameter: dict[str, object],
) -> None:
    path = _write_definitions(tmp_path, [_function({"payload": parameter})])

    with pytest.raises(
        ValueError, match="Unsupported bonus schema submit.payload",
    ):
        load_function_definitions(path, allow_nested=True)


def test_nested_function_call_result_accepts_and_rejects_values(
    tmp_path: Path,
) -> None:
    path = _write_definitions(tmp_path, [_nested_definition()])
    definitions = load_function_definitions(path, allow_nested=True)
    prompt = PromptItem(prompt="send the record")
    data = [{
        "prompt": prompt.prompt,
        "name": "submit",
        "parameters": {"payload": {"records": [{"label": "a", "score": 1.5}]}},
    }]

    calls = validate_function_calls(
        data, definitions, [prompt], allow_nested=True,
        schema_limits=SchemaLimits(),
    )

    assert calls[0].parameters == data[0]["parameters"]
    invalid = [{
        **data[0],
        "parameters": {"payload": {"records": [
            {"label": "a", "score": True},
        ]}},
    }]
    with pytest.raises(ValidationError):
        validate_function_calls(
            invalid, definitions, [prompt], allow_nested=True,
            schema_limits=SchemaLimits(),
        )


def test_nested_result_obeys_passed_resource_limits(tmp_path: Path) -> None:
    definitions = load_function_definitions(
        _write_definitions(tmp_path, [_nested_definition()]),
        allow_nested=True,
    )
    prompt = PromptItem(prompt="send")
    too_many = [{
        "prompt": "send",
        "name": "submit",
        "parameters": {"payload": {"records": [
            {"label": "a", "score": 1}, {"label": "b", "score": 2},
        ]}},
    }]

    with pytest.raises(ValidationError):
        validate_function_calls(
            too_many, definitions, [prompt], allow_nested=True,
            schema_limits=SchemaLimits(max_items=1),
        )


def test_run_settings_profile_defaults_and_local_tokenizer_guard() -> None:
    settings = RunSettings()
    assert settings.profile == "mandatory"
    assert settings.tokenizer == "sdk"

    with pytest.raises(ValidationError, match="requires --profile bonus"):
        RunSettings(profile="mandatory", tokenizer="local")

    assert RunSettings(profile="bonus", tokenizer="local").profile == "bonus"
