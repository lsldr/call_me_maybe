"""Tests for the JSON string byte DFA."""

import pytest

from src.constraints.string_dfa import StringDFA, StringState


def consume(data: bytes) -> StringState | None:
    """Return the state after consuming data, or None if it is invalid."""
    dfa = StringDFA()
    return dfa.advance(dfa.initial_state(), data)


@pytest.mark.parametrize(
    "content",
    [
        b'"',
        b'quote: \\"' + b'"',
        b'slash: \\/"',
        b'/"',
        b'\\b\\f\\n\\r\\t"',
        b'\\u0041"',
    ],
)
def test_valid_escapes_and_empty_string(content: bytes) -> None:
    dfa = StringDFA()
    state = dfa.advance(dfa.initial_state(), content)
    assert state is not None
    assert dfa.is_complete(state)


def test_surrogate_pair_may_be_split_across_tokens() -> None:
    dfa = StringDFA()
    state = dfa.initial_state()
    for chunk in (b"\\uD83D", b"\\", b"uDE00", b'"'):
        next_state = dfa.advance(state, chunk)
        assert next_state is not None
        state = next_state
    assert dfa.is_complete(state)


def test_completed_escape_states_are_canonical() -> None:
    dfa = StringDFA()
    initial = dfa.initial_state()
    escaped = dfa.advance(initial, b"\\u0041")
    assert escaped == initial

    paired = dfa.advance(initial, b"\\uD83D\\uDE00")
    assert paired == initial

    utf8 = dfa.advance(initial, b"\xc3\xa9")
    assert utf8 == initial


@pytest.mark.parametrize(
    "raw",
    [b"\\uD800\"", b"\\uDC00\"", b"\\uD800\\u0041\"", b"\\uD800x\""],
)
def test_rejects_unpaired_surrogates(raw: bytes) -> None:
    assert consume(raw) is None


@pytest.mark.parametrize(
    "raw",
    [
        b"\xc0\x80\"",
        b"\xed\xa0\x80\"",
        b"\xf4\x90\x80\x80\"",
        b"\xf5\x80\x80\x80\"",
        b"\x80\"",
    ],
)
def test_rejects_invalid_utf8(raw: bytes) -> None:
    assert consume(raw) is None


def test_utf8_can_be_split_and_quote_cannot_interrupt_it() -> None:
    dfa = StringDFA()
    state = dfa.advance(dfa.initial_state(), b"\xc3")
    assert state is not None
    assert dfa.advance(state, b'"') is None
    state = dfa.advance(state, b"\xa9")
    assert state is not None
    state = dfa.advance(state, b'"')
    assert state is not None and dfa.is_complete(state)


@pytest.mark.parametrize(
    "raw",
    [
        b"\\d\"",
        b"line\nfeed\"",
        b"abc\"x",
        b"\\u12xz\"",
        b"\xe0\x80\x80\"",
        b"\xf0\x90\x28\x80\"",
        b"",
    ],
)
def test_rejects_malformed_tokens(raw: bytes) -> None:
    assert consume(raw) is None


def test_state_is_immutable_and_advance_does_not_mutate() -> None:
    dfa = StringDFA()
    initial = dfa.initial_state()
    next_state = dfa.advance(initial, b"abc")
    assert next_state is not None
    assert initial.phase == "NORMAL"
    assert initial.utf8_remaining == 0
    with pytest.raises(Exception):
        initial.phase = "DONE"


def test_next_bytes_tracks_utf8_ranges_and_completion() -> None:
    dfa = StringDFA()
    state = dfa.advance(dfa.initial_state(), b"\xe0")
    assert state is not None
    assert dfa.next_bytes(state) == set(range(0xA0, 0xC0))
    state = dfa.initial_state()
    assert 0x22 in dfa.next_bytes(state)
