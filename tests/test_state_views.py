"""Tests for the explicit diagnostic snapshot adapters."""

from src.constraints.container import ContainerFrame, ContainerState
from src.constraints.number_boundary import NumberBoundaryState
from src.constraints.number_dfa import NumberState
from src.constraints.string_boundary import StringBoundaryState
from src.constraints.string_dfa import StringState
from src.constraints.string_value import StringValueState
from src.diagnostics.state_views import snapshot
from src.diagnostics.trace_schema import StateSnapshot


def test_string_and_wrapped_string_adapters() -> None:
    string = snapshot(StringState(phase="ESCAPE"))
    assert string.kind == "string"
    assert string.phase == "ESCAPE"

    boundary = snapshot(StringBoundaryState(
        inner=StringState(phase="UNICODE", hex_remaining=2), suffix=b"x"
    ))
    assert boundary.kind == "string-boundary"
    assert boundary.phase == "UNICODE"
    assert boundary.details["suffix_hex"] == "78"

    value = snapshot(StringValueState(phase="CONTENT"))
    assert value.kind == "string-value"
    assert value.phase == "CONTENT/NORMAL"


def test_number_and_trie_adapters() -> None:
    number = snapshot(NumberState(phase="FRAC"))
    assert number.kind == "number"
    assert number.phase == "FRAC"

    boundary = snapshot(NumberBoundaryState(
        inner=NumberState(phase="FRAC"), suffix=bytes([44]),
    ))
    assert boundary.kind == "number-boundary"
    assert boundary.phase == "FRAC"
    assert boundary.details["suffix_hex"] == "2c"

    trie = snapshot(7)
    assert trie.kind == "trie"
    assert trie.details["node"] == 7


def test_container_snapshot_and_split_utf8_pending_bytes() -> None:
    container = snapshot(ContainerState(
        stack=(ContainerFrame(node=2, phase="OBJECT_VALUE", index=1),),
        started=True,
    ))
    assert container.kind == "container"
    assert container.frames == ("node=2 phase=OBJECT_VALUE index=1 count=0",)
    assert container.details["started"] is True

    split_character = snapshot(StringState(), b"\xf0\x9f")
    assert split_character.pending_hex == "f09f"


def test_external_state_is_explicitly_unknown_and_hex_is_validated() -> None:
    unknown = snapshot(object())
    assert unknown.kind == "unknown"
    assert unknown.phase is None

    try:
        StateSnapshot(kind="string", pending_hex="not hex")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid pending bytes should be rejected")
