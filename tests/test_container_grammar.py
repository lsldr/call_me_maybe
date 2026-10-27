"""Tests for schema-constrained nested JSON byte parsing."""

import json

import pytest

from src.constraints.container import ContainerState, JSONContainerDFA
from src.models.function_def import TypeDef
from src.schemas.nested import SchemaLimits, validate_value


def schema(**values: object) -> TypeDef:
    return TypeDef.model_validate(values)


def run(
    dfa: JSONContainerDFA, value: bytes, split: int = 1,
) -> ContainerState | None:
    state: ContainerState | None = dfa.initial_state()
    for start in range(0, len(value), split):
        assert state is not None
        state = dfa.advance(state, value[start:start + split])
        if state is None:
            return None
    return state


def test_nested_value_whitespace_and_all_byte_splits() -> None:
    root = schema(type="object", properties={
        "rows": schema(type="array", items=schema(type="object", properties={
            "n": schema(type="integer"), "ok": schema(type="boolean"),
        })),
        "note": schema(type="string"),
    })
    dfa = JSONContainerDFA(schema=root)
    value = (
        '{ "rows" : [ {"n":-4,"ok":true} ], '
        '"note":"é\\uD83D\\uDE00" }'
    ).encode()
    for split in range(1, len(value) + 1):
        state = run(dfa, value, split)
        assert state is not None and dfa.is_complete(state)


@pytest.mark.parametrize("value", [b"{}", b"[]"])
def test_empty_containers(value: bytes) -> None:
    dfa = JSONContainerDFA(
        schema=(
            schema(
                type="object",
                properties={}) if value == b"{}" else schema(
                type="array",
                items=schema(
                    type="null"))))
    state = run(dfa, value)
    assert state is not None and dfa.is_complete(state)


def test_object_keys_are_exact_and_ordered() -> None:
    dfa = JSONContainerDFA(schema=schema(type="object", properties={
        "a": schema(type="integer"), "b": schema(type="integer"),
    }))
    for value in (b'{"b":1,"a":2}', b'{"a":1}', b'{"a":1,"b":2,"c":3}',
                  b'{"a":1,}'):
        assert run(dfa, value) is None


def test_unicode_and_marker_escaped_object_keys() -> None:
    marker = "<|halt|>"
    dfa = JSONContainerDFA(
        schema=schema(type="object", properties={
            "café": schema(type="integer"),
            marker: schema(type="number"),
        }),
        markers=(marker,),
    )
    assert run(dfa, '{"café":2,"\\u003c|halt|>":1.5}'.encode()) is not None
    assert run(dfa, '{"cafe\\u0301":2,"\\u003c|halt|>":1.5}'.encode()) is None


def test_float_number_inside_array() -> None:
    dfa = JSONContainerDFA(
        schema=schema(type="array", items=schema(type="number")),
    )
    state = run(dfa, b"[1.25,-0.5,0]")
    assert state is not None and dfa.is_complete(state)


@pytest.mark.parametrize("value", [
    b'{"x":null}', b'{"x":1.0}', b'{"x":01}', b'{"x":-}',
    b'{"x":truex}', b'{"x":False}', b'{"x":"\\uD800"}',
    b'{"x":"\xff"}', b'{"x":"\\uDC00"}',
])
def test_invalid_scalar_and_json_syntax(value: bytes) -> None:
    dfa = JSONContainerDFA(schema=schema(type="object", properties={
        "x": schema(type="integer"),
    }))
    assert run(dfa, value) is None


def test_strings_and_boolean_types() -> None:
    dfa = JSONContainerDFA(schema=schema(
        type="array", items=schema(type="boolean")))
    assert run(dfa, b"[true,false]") is not None
    assert run(dfa, b"[1]") is None


@pytest.mark.parametrize("value", [b"[talse]", b"[frue]"])
def test_boolean_literal_cannot_switch_spelling_mid_token(
        value: bytes) -> None:
    dfa = JSONContainerDFA(
        schema=schema(type="array", items=schema(type="boolean")),
    )
    assert run(dfa, value) is None


def test_nested_boolean_literal_cannot_switch_spelling() -> None:
    dfa = JSONContainerDFA(schema=schema(
        type="array",
        items=schema(type="object", properties={
            "ok": schema(type="boolean"),
        }),
    ))
    assert run(dfa, b'[{"ok":talse}]') is None
    assert run(dfa, b'[{"ok":frue}]') is None


@pytest.mark.parametrize(("root", "candidates"), [
    (
        schema(type="array", items=schema(type="boolean")),
        (b"[]", b"[true]", b"[false,true]", b"[talse]", b"[frue]",
         b"[true,]", b"[true,,false]", b"[1]", b"[null]"),
    ),
    (
        schema(type="object", properties={
            "flags": schema(type="array", items=schema(type="boolean")),
            "ready": schema(type="boolean"),
        }),
        (b'{"flags":[],"ready":true}',
         b'{"flags":[false,true],"ready":false}',
         b'{"flags":[talse],"ready":true}',
         b'{"flags":[true,],"ready":true}',
         b'{"ready":true,"flags":[]}',
         b'{"flags":[],"ready":1}',
         b'{"flags":[],"ready":true,}'),
    ),
])
def test_deterministic_boolean_grammar_corpus_matches_json_oracle(
    root: TypeDef, candidates: tuple[bytes, ...],
) -> None:
    dfa = JSONContainerDFA(schema=root)

    def declared_order(value: object, node: TypeDef) -> bool:
        if node.type == "array":
            assert node.items is not None and isinstance(value, list)
            return all(declared_order(item, node.items) for item in value)
        if node.type == "object":
            assert isinstance(value, dict)
            properties = node.properties or {}
            return (list(value) == list(properties) and all(
                declared_order(value[key], child)
                for key, child in properties.items()
            ))
        return True

    for candidate in candidates:
        expected = False
        try:
            value = json.loads(candidate)
            validate_value(value, root)
            expected = declared_order(value, root)
        except (ValueError, TypeError, UnicodeDecodeError):
            pass
        state = run(dfa, candidate)
        actual = state is not None and dfa.is_complete(state)
        assert actual is expected, candidate


def test_merged_following_is_prefix_matched_without_dropping_bytes() -> None:
    dfa = JSONContainerDFA(
        schema=schema(type="array", items=schema(type="integer")),
        following=b"\nEND",
    )
    state = dfa.advance(dfa.initial_state(), b"[1]\nE")
    assert state is not None and state.suffix == b"\nE" and dfa.is_complete(
        state)
    assert dfa.advance(state, b"N") is None
    assert run(dfa, b"[1]\nX") is None
    assert run(dfa, b"[1]junk") is None


def test_array_bounds_and_model_choice() -> None:
    dfa = JSONContainerDFA(schema=schema(
        type="array", items=schema(type="integer"), minItems=1, maxItems=2,
    ))
    state: ContainerState | None = dfa.initial_state()
    for byte in b"[1":
        assert state is not None
        state = dfa.advance(state, bytes([byte]))
        assert state is not None
    assert state is not None
    assert ord("]") in dfa.next_bytes(state)
    assert ord(",") in dfa.next_bytes(state)
    assert run(dfa, b"[]") is None
    assert run(dfa, b"[1,2,3]") is None


def test_schema_depth_and_root_validation() -> None:
    deep = schema(type="array", items=schema(
        type="array", items=schema(type="null")))
    with pytest.raises(ValueError):
        JSONContainerDFA(schema=deep, limits=SchemaLimits(max_depth=1))
    with pytest.raises(ValueError):
        JSONContainerDFA(schema=schema(type="integer"))


def test_completion_in_one_token_and_identity() -> None:
    dfa = JSONContainerDFA(schema=schema(type="object", properties={
        "x": schema(type="array", items=schema(type="null")),
    }), following=b"!")
    state = dfa.advance(dfa.initial_state(), b'{"x":[null]}!')
    assert state is not None and dfa.is_complete(state)
    assert state.suffix == b"!"
    assert dfa.identity == JSONContainerDFA(
        schema=dfa.root_schema, following=b"!").identity


def test_identity_preserves_declared_property_order() -> None:
    left = schema(type="object", properties={
        "a": schema(type="integer"), "b": schema(type="integer"),
    })
    right = schema(type="object", properties={
        "b": schema(type="integer"), "a": schema(type="integer"),
    })
    assert JSONContainerDFA(schema=left).identity != JSONContainerDFA(
        schema=right).identity


def test_schema_is_snapshotted_at_construction() -> None:
    original = schema(type="object", properties={
        "x": schema(type="integer"),
    })
    dfa = JSONContainerDFA(schema=original)
    assert original.properties is not None
    original.properties["x"] = schema(type="string")
    state = run(dfa, b'{"x":1}')
    assert state is not None and dfa.is_complete(state)
