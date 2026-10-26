"""Tests for JSON string boundaries merged into tokenizer tokens."""

from src.constraints.string_boundary import StringBoundaryDFA
from src.constraints.string_dfa import StringDFA


def test_boundary_suffix_records_exact_following_prefix() -> None:
    """The token may close a string and consume a following prefix."""
    following = b', "next": '
    dfa = StringBoundaryDFA(following=following)
    state = dfa.advance(dfa.initial_state(), b'x", "')
    assert state is not None
    assert dfa.is_complete(state)
    assert state.suffix == b', "'


def test_base_string_dfa_still_rejects_bytes_after_closing_quote() -> None:
    """The standalone string recognizer keeps its strict token boundary."""
    assert StringDFA().advance(StringDFA().initial_state(), b'x", "') is None


def test_wrong_following_content_and_overlong_suffix_are_rejected() -> None:
    """Only an exact prefix within the configured boundary is valid."""
    dfa = StringBoundaryDFA(following=b', "next": ')
    assert dfa.advance(dfa.initial_state(), b'x", "oops') is None
    assert dfa.advance(dfa.initial_state(), b'x";') is None
    assert dfa.advance(dfa.initial_state(), b'x", "next": 1') is None


def test_split_utf8_and_escaped_quote_stay_inside_the_string() -> None:
    """Boundary handling preserves the inner DFA's UTF-8 and escape checks."""
    dfa = StringBoundaryDFA(following=b", ")
    state = dfa.advance(dfa.initial_state(), b"\xc3")
    assert state is not None and not dfa.is_complete(state)
    state = dfa.advance(state, b'\xa9\\"')
    assert state is not None and not dfa.is_complete(state)
    state = dfa.advance(state, b'"')
    assert state is not None and dfa.is_complete(state)
    assert state.suffix == b""


def test_quote_with_no_following_bytes_completes_normally() -> None:
    """A normal quote-only final token has an empty recorded suffix."""
    dfa = StringBoundaryDFA(following=b", ")
    state = dfa.advance(dfa.initial_state(), b'value"')
    assert state is not None and dfa.is_complete(state)
    assert state.suffix == b""


def test_token_can_consume_multiple_closing_braces_and_newline() -> None:
    """Every byte after the quote is retained when it matches the boundary."""
    following = b"}}\n"
    dfa = StringBoundaryDFA(following=following)
    state = dfa.advance(dfa.initial_state(), b'x"}}\n')
    assert state is not None and dfa.is_complete(state)
    assert state.suffix == following


def test_empty_candidates_and_bytes_after_completion_are_rejected() -> None:
    """Completed boundary tokens cannot be extended again."""
    dfa = StringBoundaryDFA(following=b", ")
    initial = dfa.initial_state()
    assert dfa.advance(initial, b"") is None
    completed = dfa.advance(initial, b'x",')
    assert completed is not None
    assert dfa.advance(completed, b" ") is None
