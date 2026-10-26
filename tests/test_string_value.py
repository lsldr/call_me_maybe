"""Tests for full JSON string values with optional boundary bytes."""

import pytest

from src.constraints.string_value import StringValueDFA, StringValueState


def test_split_opening_content_and_boundary_prefix() -> None:
    """Opening quote, UTF-8 content, and boundary may arrive separately."""
    dfa = StringValueDFA(following=b"}}\n")
    state = dfa.initial_state()
    for token in (b" ", b'"', b"\xc3", b"\xa9", b'"}}'):
        next_state = dfa.advance(state, token)
        assert next_state is not None
        state = next_state
    assert dfa.is_complete(state)
    assert state.inner.suffix == b"}}"


def test_merged_opening_content_closing_and_following_suffix() -> None:
    """One token may contain the whole value and a boundary prefix."""
    dfa = StringValueDFA(following=b', "next": "')
    raw = b' "hello", "next": "'
    state = dfa.advance(dfa.initial_state(), raw)
    assert state is not None
    assert dfa.is_complete(state)
    assert state.inner.suffix == b', "next": "'


def test_empty_string_and_opening_quote_only_prefix() -> None:
    """The opening quote can be an incomplete token or an empty value."""
    dfa = StringValueDFA(following=b"}}\n")
    opened = dfa.advance(dfa.initial_state(), b'"')
    assert opened is not None
    assert not dfa.is_complete(opened)
    complete = dfa.advance(opened, b'"}}')
    assert complete is not None
    assert dfa.is_complete(complete)
    assert complete.inner.suffix == b"}}"


def test_optional_space_is_exactly_one_ascii_space() -> None:
    """One leading space is valid while repeated or tab whitespace is not."""
    dfa = StringValueDFA(following=b"}}\n")
    accepted = dfa.advance(dfa.initial_state(), b' "x"}}\n')
    assert accepted is not None and dfa.is_complete(accepted)
    for raw in (b'  "x"}}\n', b'\t"x"}}\n', b' \t"x"}}\n'):
        assert dfa.advance(dfa.initial_state(), raw) is None


@pytest.mark.parametrize(
    "raw",
    [
        b"x\"}}\n",
        b'"\\q"}}\n',
        b'"x";',
        b'"x"}}\nextra',
    ],
)
def test_missing_open_quote_and_invalid_content_or_boundary_are_rejected(
    raw: bytes,
) -> None:
    """The complete-value DFA checks both string spelling and exact suffix."""
    dfa = StringValueDFA(following=b"}}\n")
    assert dfa.advance(dfa.initial_state(), raw) is None


def test_completed_state_cannot_consume_more_bytes() -> None:
    """A completed string value is terminal."""
    dfa = StringValueDFA(following=b"}}\n")
    state = dfa.advance(dfa.initial_state(), b'"x"}}')
    assert state is not None and dfa.is_complete(state)
    assert dfa.advance(state, b"\n") is None
    assert dfa.next_bytes(state) == set()


def test_empty_input_and_state_trials_are_pure() -> None:
    """Empty trials reject and immutable states stay unchanged."""
    dfa = StringValueDFA(following=b"}}\n")
    initial = dfa.initial_state()
    assert dfa.advance(initial, b"") is None
    next_state = dfa.advance(initial, b' "text')
    assert next_state is not None
    assert initial == StringValueState()
    assert initial.phase == "START"
