"""Validation for the deliberately small nested-schema subset we support."""

import json
import math
from typing import TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field

from src.models.function_def import TypeDef


class SchemaLimits(BaseModel):
    """Resource limits applied to supported schemas and values."""

    model_config = ConfigDict(frozen=True, strict=True)

    max_depth: int = Field(default=8, ge=1)
    max_items: int = Field(default=32, ge=0)
    max_bytes: int = Field(default=4096, ge=1)


_SCALARS = {"string", "number", "integer", "boolean", "null"}
_ANNOTATIONS = {"title", "examples", "default"}
_STRUCTURAL = {"items", "properties", "required", "additionalProperties",
               "minItems", "maxItems"}
_Container: TypeAlias = dict[str, object] | list[object]


def _limits(limits: SchemaLimits | None) -> SchemaLimits:
    return limits if limits is not None else SchemaLimits()


def _extras(schema: TypeDef) -> dict[str, object]:
    return dict(schema.model_extra or {})


def _validate_schema(
    schema: TypeDef, limits: SchemaLimits, depth: int
) -> None:
    if not isinstance(schema, TypeDef):
        raise ValueError("schema nodes must be TypeDef instances")
    extras = _extras(schema)
    unknown = set(extras) - _STRUCTURAL - _ANNOTATIONS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unsupported schema keyword(s): {names}")

    kind = schema.type
    if kind in _SCALARS:
        if set(extras) - _ANNOTATIONS:
            raise ValueError(f"unsupported validation keyword(s) for {kind}")
        if schema.items is not None or schema.properties is not None:
            raise ValueError(
                f"{kind} schemas cannot declare items or properties"
            )
        return

    if kind == "array":
        if set(extras) - _ANNOTATIONS - {"minItems", "maxItems"}:
            raise ValueError("unsupported validation keyword(s) for array")
        if schema.properties is not None:
            raise ValueError("array schemas cannot declare properties")
        if schema.items is None:
            raise ValueError("array schemas require homogeneous items")
        if depth > limits.max_depth:
            raise ValueError("schema exceeds maximum container depth")
        minimum = extras.get("minItems", 0)
        maximum = extras.get("maxItems", limits.max_items)
        _check_array_bound(minimum, "minItems")
        _check_array_bound(maximum, "maxItems")
        minimum = cast(int, minimum)
        maximum = cast(int, maximum)
        if minimum > min(maximum, limits.max_items):
            raise ValueError(
                "minItems exceeds the effective maximum array size"
            )
        _validate_schema(schema.items, limits, depth + 1)
        return

    if kind == "object":
        if set(extras) - _ANNOTATIONS - {"required", "additionalProperties"}:
            raise ValueError("unsupported validation keyword(s) for object")
        if schema.items is not None:
            raise ValueError("object schemas cannot declare items")
        if schema.properties is None:
            raise ValueError(
                "object schemas require an explicit properties mapping"
            )
        if depth > limits.max_depth:
            raise ValueError("schema exceeds maximum container depth")
        required = extras.get("required")
        keys = list(schema.properties)
        if required is not None:
            if (not isinstance(required, list)
                    or any(type(key) is not str for key in required)
                    or len(set(required)) != len(required)
                    or set(required) != set(keys)):
                raise ValueError(
                    "required must list every declared property exactly once"
                )
        additional = extras.get("additionalProperties")
        if "additionalProperties" in extras and additional is not False:
            raise ValueError(
                "additionalProperties must be false when specified"
            )
        for child in schema.properties.values():
            _validate_schema(child, limits, depth + 1)
        return

    raise ValueError(f"unsupported schema type: {kind}")


def _check_array_bound(bound: object, name: str) -> None:
    if type(bound) is not int or bound < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def validate_schema(
    schema: TypeDef, limits: SchemaLimits | None = None
) -> None:
    """Reject schemas outside the supported nested subset."""
    chosen = _limits(limits)
    depth = 1 if schema.type in {"array", "object"} else 0
    _validate_schema(schema, chosen, depth)


def array_bounds(
    schema: TypeDef, limits: SchemaLimits | None = None
) -> tuple[int, int]:
    """Return the effective inclusive item-count bounds for an array schema."""
    chosen = _limits(limits)
    validate_schema(schema, chosen)
    if schema.type != "array":
        raise ValueError("array_bounds requires an array schema")
    extras = _extras(schema)
    minimum_value = extras.get("minItems", 0)
    maximum_value = extras.get("maxItems", chosen.max_items)
    _check_array_bound(minimum_value, "minItems")
    _check_array_bound(maximum_value, "maxItems")
    minimum = cast(int, minimum_value)
    maximum = min(cast(int, maximum_value), chosen.max_items)
    return minimum, maximum


def _validate_value(value: object, schema: TypeDef, limits: SchemaLimits,
                    depth: int) -> None:
    kind = schema.type
    if kind in _SCALARS:
        if kind == "string":
            valid = type(value) is str and not any(
                0xD800 <= ord(char) <= 0xDFFF for char in value
            )
        elif kind == "number":
            valid = type(value) in (int, float) and (
                type(value) is int or math.isfinite(cast(float, value))
            )
        elif kind == "integer":
            valid = type(value) is int
        elif kind == "boolean":
            valid = type(value) is bool
        else:
            valid = value is None
        if not valid:
            raise ValueError(f"value does not match {kind} schema")
        return

    if depth > limits.max_depth:
        raise ValueError("value exceeds maximum container depth")
    if kind == "array":
        if type(value) is not list:
            raise ValueError("value does not match array schema")
        minimum, maximum = array_bounds(schema, limits)
        if not minimum <= len(value) <= maximum:
            raise ValueError("array length is outside the permitted bounds")
        assert schema.items is not None
        for item in value:
            _validate_value(item, schema.items, limits, depth + 1)
        return

    if kind == "object":
        if type(value) is not dict:
            raise ValueError("value does not match object schema")
        properties = schema.properties or {}
        if set(value) != set(properties):
            raise ValueError(
                "object keys must exactly match declared properties"
            )
        for key, child in properties.items():
            _validate_value(value[key], child, limits, depth + 1)
        return
    raise ValueError(f"unsupported schema type: {kind}")


def validate_value(value: object, schema: TypeDef,
                   limits: SchemaLimits | None = None) -> None:
    """Validate a Python value and enforce compact UTF-8 JSON byte limits."""
    chosen = _limits(limits)
    validate_schema(schema, chosen)
    depth = 1 if schema.type in {"array", "object"} else 0
    _validate_value(value, schema, chosen, depth)
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError(
            "value cannot be encoded as compact UTF-8 JSON"
        ) from error
    if len(encoded) > chosen.max_bytes:
        raise ValueError("value exceeds maximum serialized byte size")
