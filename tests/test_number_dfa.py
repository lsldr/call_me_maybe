"""Tests for the decimal number byte-state machine."""

from typing import Literal

import pytest
from pydantic import ValidationError

from src.constraints.number_dfa import NumberDFA, NumberState


@pytest.mark.parametrize(
    ("raw", "delimiter", "integer"),
    [
        (b"0", "}", False),
        (b"-0", "}", False),
        (b"-12.5", "}", False),
        (b"12345", ",", False),
        (b"42", "}", True),
    ],
)
def test_valid_values_and_explicit_delimiter(
    raw: bytes, delimiter: Literal[",", "}"], integer: bool
) -> None:
    """A complete value becomes DONE only when its delimiter is consumed."""
    dfa = NumberDFA(integer=integer, delimiter=delimiter)
    state = dfa.advance(dfa.initial_state(), raw)
    assert state is not None
    assert not dfa.is_complete(state)
    finished = dfa.advance(state, delimiter.encode())
    assert finished == NumberState(phase="DONE")
    assert dfa.is_complete(finished)
    assert dfa.advance(finished, b"x") is None


@pytest.mark.parametrize("parts", [(b"-", b"12", b".", b"5"), (b"1", b"2.5")])
def test_split_tokens(parts: tuple[bytes, ...]) -> None:
    """A number may be consumed over several candidate tokens."""
    dfa = NumberDFA()
    state = dfa.initial_state()
    for part in parts:
        next_state = dfa.advance(state, part)
        assert next_state is not None
        state = next_state
    assert state.phase in {"INT", "FRAC"}


@pytest.mark.parametrize("raw", [b"00", b"-01", b"1e3", b" 1", b"1 "])
def test_invalid_number_forms_are_rejected(raw: bytes) -> None:
    """Leading zeros, exponents, and whitespace are outside this subset."""
    assert NumberDFA().advance(NumberState(), raw) is None


def test_integer_mode_rejects_decimal_point() -> None:
    """Integer mode has no decimal transition."""
    assert NumberDFA(integer=True).advance(NumberState(), b"1.") is None
    assert ord(".") not in NumberDFA(integer=True).next_bytes(
        NumberState(phase="INT")
    )


@pytest.mark.parametrize("prefix", [b"-", b"12."])
def test_incomplete_prefixes_are_valid_but_not_complete(prefix: bytes) -> None:
    """A prefix may be extendable without representing a complete value."""
    dfa = NumberDFA()
    state = dfa.advance(dfa.initial_state(), prefix)
    assert state is not None
    assert not dfa.is_complete(state)
    assert dfa.advance(state, b"}") is None


def test_delimiter_must_match_and_be_last_byte() -> None:
    """Only the configured delimiter can finish a value, as the final byte."""
    dfa = NumberDFA(delimiter=",")
    assert dfa.advance(NumberState(), b"12}") is None
    assert dfa.advance(NumberState(), b"12,x") is None
    done = dfa.advance(NumberState(), b"12,")
    assert done is not None and dfa.is_complete(done)


def test_empty_input_and_delimiter_suffix_are_rejected() -> None:
    """Empty candidates and any bytes after completion are invalid."""
    dfa = NumberDFA()
    assert dfa.advance(NumberState(), b"") is None
    done = dfa.advance(NumberState(), b"12}x")
    assert done is None


def test_all_phases_have_expected_next_bytes() -> None:
    """The six numeric phases expose their legal continuation bytes."""
    dfa = NumberDFA()
    assert ord("-") in dfa.next_bytes(NumberState(phase="START"))
    assert ord("0") in dfa.next_bytes(NumberState(phase="SIGN"))
    assert ord(".") in dfa.next_bytes(NumberState(phase="ZERO"))
    assert ord("2") in dfa.next_bytes(NumberState(phase="INT"))
    assert ord("3") in dfa.next_bytes(NumberState(phase="DOT"))
    assert ord("}") in dfa.next_bytes(NumberState(phase="FRAC"))
    assert dfa.next_bytes(NumberState(phase="DONE")) == set()


def test_trial_is_pure_and_models_are_immutable() -> None:
    """A failed or successful trial never changes its supplied state."""
    dfa = NumberDFA()
    original = NumberState()
    assert dfa.advance(original, b"-1") is not None
    assert original.phase == "START"
    with pytest.raises(ValidationError):
        setattr(original, "phase", "INT")


def test_optional_leading_space_accepts_signed_decimal_and_delimiter() -> None:
    """The opt-in initial space may precede one signed decimal value."""
    dfa = NumberDFA(allow_leading_space=True)
    state = dfa.advance(dfa.initial_state(), b" -12.5}")
    assert state is not None and dfa.is_complete(state)
    assert ord(" ") in dfa.next_bytes(dfa.initial_state())
    assert dfa.next_bytes(NumberState(phase="SPACE")) == {
        ord("-"), ord("0"), *range(ord("1"), ord("9") + 1)
    }


def test_leading_space_and_number_can_be_split_across_tokens() -> None:
    """The initial space may be split from the sign and number tokens."""
    dfa = NumberDFA(allow_leading_space=True)
    state = dfa.initial_state()
    for token in (b" ", b"-", b"12.5}"):
        next_state = dfa.advance(state, token)
        assert next_state is not None
        state = next_state
    assert dfa.is_complete(state)


@pytest.mark.parametrize("raw", [b"  1", b" \t1", b" 1 ", b" 1 2"])
def test_optional_leading_space_rejects_other_whitespace(raw: bytes) -> None:
    """At most one initial ASCII space is accepted."""
    assert NumberDFA(allow_leading_space=True).advance(
        NumberState(), raw
    ) is None


def test_optional_leading_space_preserves_integer_mode_and_default() -> None:
    """Integer mode accepts the optional prefix but no decimal point."""
    assert NumberDFA(integer=True, allow_leading_space=True).advance(
        NumberState(), b" 42}"
    ) == NumberState(phase="DONE")
    assert NumberDFA(integer=True, allow_leading_space=True).advance(
        NumberState(), b" 4.2}"
    ) is None
    assert NumberDFA().advance(NumberState(), b" 42}") is None
    assert ord(" ") not in NumberDFA().next_bytes(NumberState())
