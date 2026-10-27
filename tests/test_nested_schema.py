"""Tests for nested schema validation."""

import math

import pytest
from pydantic import ValidationError

from src.models.function_def import TypeDef
from src.schemas.nested import (
    SchemaLimits,
    array_bounds,
    validate_schema,
    validate_value,
)


def make_schema(**data: object) -> TypeDef:
    """Build a TypeDef with metadata outside its statically declared fields."""
    return TypeDef.model_validate(data)


def test_nested_objects_and_arrays_validate() -> None:
    schema = TypeDef(
        type="object",
        properties={
            "rows": TypeDef(type="array", items=TypeDef(
                type="object", properties={"count": TypeDef(type="integer")},
            )),
        },
    )
    value = {"rows": [{"count": 2}, {"count": 0}]}
    validate_schema(schema)
    validate_value(value, schema)


def test_empty_object_and_array_are_allowed() -> None:
    validate_value({}, TypeDef(type="object", properties={}))
    validate_value([], TypeDef(type="array", items=TypeDef(type="null")))


def test_exact_required_keys_and_metadata_preserved() -> None:
    schema = make_schema(
        type="object", properties={"x": TypeDef(type="string")},
        required=["x"], additionalProperties=False,
        title="Input", examples=[{"x": "ok"}],
    )
    validate_schema(schema)
    assert schema.model_extra == {
        "required": ["x"], "additionalProperties": False,
        "title": "Input", "examples": [{"x": "ok"}],
    }
    for required in (["x", "x"], [], ["other"]):
        with pytest.raises(ValueError):
            schema = make_schema(
                type="object", properties={"x": TypeDef(type="string")},
                required=required,
            )
            validate_schema(schema)


@pytest.mark.parametrize("value,schema", [
    (True, TypeDef(type="integer")),
    (1, TypeDef(type="boolean")),
    (1.5, TypeDef(type="integer")),
    (math.inf, TypeDef(type="number")),
    ("\ud800", TypeDef(type="string")),
    ({"x": 1}, TypeDef(
        type="object", properties={"y": TypeDef(type="integer")},
    )),
    ([True], TypeDef(type="array", items=TypeDef(type="integer"))),
])
def test_bad_values_rejected(value: object, schema: TypeDef) -> None:
    with pytest.raises(ValueError):
        validate_value(value, schema)


def test_array_bounds_enforced_and_capped() -> None:
    schema = make_schema(
        type="array", items=TypeDef(type="integer"), minItems=1, maxItems=50,
    )
    assert array_bounds(schema, SchemaLimits(max_items=3)) == (1, 3)
    with pytest.raises(ValueError):
        validate_value([], schema)
    with pytest.raises(ValueError):
        validate_value([1, 2, 3, 4], schema, SchemaLimits(max_items=3))
    with pytest.raises(ValueError):
        schema = make_schema(
            type="array", items=TypeDef(type="null"), minItems=4,
        )
        validate_schema(schema, SchemaLimits(max_items=3))


def test_depth_item_and_serialized_byte_limits() -> None:
    deep = TypeDef(type="array", items=TypeDef(
        type="array", items=TypeDef(type="string"),
    ))
    with pytest.raises(ValueError):
        validate_value([["x"]], deep, SchemaLimits(max_depth=1))
    with pytest.raises(ValueError):
        schema = TypeDef(type="array", items=TypeDef(type="integer"))
        validate_value([1, 2], schema, SchemaLimits(max_items=1))
    with pytest.raises(ValueError):
        validate_value("é", TypeDef(type="string"), SchemaLimits(max_bytes=3))


@pytest.mark.parametrize("schema", [
    make_schema(type="string", enum=["a"]),
    make_schema(
        type="object", properties={"x": TypeDef(type="string")},
        required=["x"], additionalProperties=True,
    ),
    make_schema(type="array", items=TypeDef(type="string"), minItems=True),
])
def test_unsupported_or_invalid_keywords_rejected(schema: TypeDef) -> None:
    with pytest.raises(ValueError):
        validate_schema(schema)


def test_optional_union_and_reference_shapes_rejected() -> None:
    optional = make_schema(
        type="object", properties={"x": TypeDef(type="string")}, required=[],
    )
    union = make_schema(
        type="string", anyOf=[{"type": "string"}, {"type": "null"}],
    )
    reference = make_schema(type="string", **{"$ref": "#/defs/X"})
    for schema in (optional, union, reference):
        with pytest.raises(ValueError):
            validate_schema(schema)


def test_invalid_limit_config_rejected() -> None:
    with pytest.raises(ValidationError):
        SchemaLimits(max_depth=0)
    with pytest.raises(ValidationError):
        SchemaLimits(max_items=True)
