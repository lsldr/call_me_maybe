"""Tests for the opt-in number boundary adapter."""

from pydantic import ValidationError
import pytest

from src.constraints.number_boundary import (
    NumberBoundaryDFA,
    NumberBoundaryState,
)
from src.constraints.number_dfa import NumberDFA


def test_no_suffix_preserves_original_number_dfa_behavior() -> None:
    boundary = NumberBoundaryDFA(number=NumberDFA(delimiter="}"))
    state = boundary.initial_state()
    completed = boundary.advance(state, b"12}")
    assert completed is not None
    assert boundary.is_complete(completed)
    assert completed.suffix == b""
    assert boundary.advance(state, b"12}}\n") is None


def test_combined_and_split_number_delimiter_suffix() -> None:
    boundary = NumberBoundaryDFA(
        number=NumberDFA(delimiter="}"), following=b"}\n",
    )
    state = boundary.initial_state()
    split = boundary.advance(state, b"1")
    assert split is not None
    split = boundary.advance(split, b"2")
    assert split is not None
    combined = boundary.advance(split, b"}}\n")
    assert combined is not None
    assert boundary.is_complete(combined)
    assert combined.suffix == b"}\n"

    whole = boundary.advance(state, b"12}}\n")
    assert whole is not None
    assert whole.suffix == b"}\n"


def test_wrong_or_excessive_suffix_is_rejected() -> None:
    boundary = NumberBoundaryDFA(
        number=NumberDFA(delimiter="}"), following=b"}\n",
    )
    state = boundary.initial_state()
    assert boundary.advance(state, b"12}x") is None
    assert boundary.advance(state, b"12}}\nextra") is None
    assert boundary.advance(state, b"12,}\n") is None


def test_integer_and_fractional_number_modes() -> None:
    integer = NumberBoundaryDFA(
        number=NumberDFA(integer=True, delimiter=","), following=b" ",
    )
    integer_state = integer.advance(integer.initial_state(), b"37, ")
    assert integer_state is not None
    assert integer_state.suffix == b" "

    decimal = NumberBoundaryDFA(
        number=NumberDFA(delimiter="}"), following=b"}\n",
    )
    decimal_state = decimal.advance(decimal.initial_state(), b"-12.5}}\n")
    assert decimal_state is not None
    assert decimal_state.suffix == b"}\n"


def test_incomplete_sign_or_fraction_cannot_close() -> None:
    dfa = NumberBoundaryDFA(
        number=NumberDFA(delimiter="}"), following=b"}\n",
    )
    state = dfa.initial_state()
    sign = dfa.advance(state, b"-")
    assert sign is not None
    assert dfa.advance(sign, b"}\n") is None

    decimal = dfa.advance(state, b"1.")
    assert decimal is not None
    assert dfa.advance(decimal, b"}\n") is None


def test_state_is_immutable_and_complete_state_rejects_consumption() -> None:
    dfa = NumberBoundaryDFA(
        number=NumberDFA(delimiter="}"), following=b"}\n",
    )
    initial = dfa.initial_state()
    advanced = dfa.advance(initial, b"12}}\n")
    assert advanced is not None
    assert initial.inner.phase == "START"
    assert initial.suffix == b""
    assert dfa.next_bytes(advanced) == set()
    assert dfa.advance(advanced, b"x") is None
    with pytest.raises(ValidationError):
        advanced.suffix = b"changed"
    assert NumberBoundaryState().inner.phase == "START"
