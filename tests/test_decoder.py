"""Behavioral tests for constrained decoding without a real model."""

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from numpy.typing import NDArray

from src.constraints.set_cache import AllowedSetCache
from src.decoding.decoder import Decoder
from src.decoding.generators import (
    generate_boolean,
    generate_name,
    generate_number,
    generate_string,
)
from src.vocab.byte_table import build_byte_to_unicode
from src.vocab.vocab_table import VocabTable


BYTE_TO_UNICODE = build_byte_to_unicode()


def make_decoder(raw_tokens: list[bytes], **limits: Any) -> Decoder:
    """Build a tiny vocabulary whose token spellings encode raw bytes."""
    spellings = [
        "".join(BYTE_TO_UNICODE[byte] for byte in raw) for raw in raw_tokens
    ]
    return Decoder(
        vocab=VocabTable(
            vocab={
                spelling: token_id
                for token_id, spelling in enumerate(spellings)
            },
            tokenizer={},
            n_logits=len(raw_tokens),
        ),
        **limits,
    )


def scripted_logits(decoder: Decoder, wanted: list[bytes], *,
                    illegal_first: bytes | None = None
                    ) -> tuple[Callable[[list[int]], NDArray[np.float32]],
                               list[list[int]]]:
    """Return a callable that scores each requested token highest by turn."""
    ids_by_raw = {
        raw: token_id for token_id, raw in enumerate(decoder.vocab.id_to_bytes)
        if raw is not None
    }
    calls: list[list[int]] = []

    def logits(context: list[int]) -> NDArray[np.float32]:
        calls.append(list(context))
        step = len(calls) - 1
        result = np.zeros(decoder.vocab.n_logits, dtype=np.float32)
        result[ids_by_raw[wanted[step]]] = 10.0
        if illegal_first is not None and step == 0:
            result[ids_by_raw[illegal_first]] = 100.0
        return result

    return logits, calls


def test_illegal_highest_score_is_masked_and_history_is_exact() -> None:
    tokens = [b"\n", b"y", b"\"", b"\"x", b"\"y"]
    decoder = make_decoder(tokens)
    logits, calls = scripted_logits(
        decoder, [b"y", b'"'], illegal_first=b"\n"
    )
    original = [7, 8]
    result = generate_string(decoder, original, logits)
    assert result.value == "y"
    assert result.fragment == b'y"'
    assert result.context_ids == [7, 8, tokens.index(b"y"), tokens.index(b'"')]
    assert original == [7, 8]
    assert calls == [[7, 8], [7, 8, tokens.index(b"y")]]


def test_names_with_shared_prefixes_unicode_and_quotes() -> None:
    names = ["run", 'run-é"', "runway"]
    token_raw = [b"run", b"-", "é".encode(), b"\\", b'"', b"x", b"y"]
    decoder = make_decoder(token_raw)
    logits, _ = scripted_logits(
        decoder, [b"run", b"-", "é".encode(), b"\\", b'"', b'"']
    )
    result = generate_name(decoder, [1], logits, names)
    assert result.value == 'run-é"'
    assert result.fragment == b"run-\xc3\xa9\\\"\""


def test_string_accepts_split_utf8_escape_and_combined_closing_token() -> None:
    raw = [b"\xc3", b"\xa9", b"\\u", b"0041", b"\"", b"\xc3\xa9\\u0041\""]
    decoder = make_decoder(raw)
    logits, _ = scripted_logits(
        decoder, [b"\xc3", b"\xa9", b"\\u", b"0041", b"\""]
    )
    result = generate_string(decoder, [3], logits)
    assert result.value == "éA"
    assert result.fragment == b"\xc3\xa9\\u0041\""

    combined = make_decoder([b"hello\""])
    one, _ = scripted_logits(combined, [b"hello\""])
    assert generate_string(combined, [3], one).value == "hello"


def test_number_delimiter_is_preserved_and_integer_overflow_fails() -> None:
    decoder = make_decoder([b"12", b",", b"}"])
    logits, _ = scripted_logits(decoder, [b"12", b","])
    result = generate_number(decoder, [0], logits, integer=True, delimiter=",")
    assert result.value == 12
    assert result.fragment == b"12,"
    assert result.consumed_delimiter == ","

    huge = b"9" * 400
    overflow_decoder = make_decoder([huge, b"}"])
    overflow_logits, _ = scripted_logits(overflow_decoder, [huge, b"}"])
    with pytest.raises(ValueError, match="finite"):
        generate_number(overflow_decoder, [0], overflow_logits)


def test_false_is_decoded_as_false() -> None:
    decoder = make_decoder([b" false", b" true"])
    logits, _ = scripted_logits(decoder, [b" false"])
    result = generate_boolean(decoder, [2], logits)
    assert result.value is False
    assert result.fragment == b" false"


def test_failure_caps_and_absent_candidate() -> None:
    tokens = [b"a", b"b", b'"']
    token_limited = make_decoder(tokens, max_new_tokens=1)
    logits, _ = scripted_logits(token_limited, [b"a"])
    with pytest.raises(ValueError, match="token limit"):
        generate_string(token_limited, [0], logits)

    byte_limited = make_decoder([b"long\""], max_value_bytes=2)
    logits, _ = scripted_logits(byte_limited, [b"long\""])
    with pytest.raises(ValueError, match="byte limit"):
        generate_string(byte_limited, [0], logits)

    context_limited = make_decoder([b'"'], max_context_tokens=1)
    logits, _ = scripted_logits(context_limited, [b'"'])
    with pytest.raises(ValueError, match="Context"):
        generate_string(context_limited, [0, 1], logits)

    no_quote = make_decoder([b"\\uD800"])
    logits, _ = scripted_logits(no_quote, [b"\\uD800"])
    with pytest.raises(ValueError, match="No allowed token continuation"):
        generate_string(no_quote, [0], logits)


def test_logits_dimension_mismatch_and_empty_context_fail() -> None:
    decoder = make_decoder([b'"'])
    with pytest.raises(ValueError, match="dimension"):
        generate_string(decoder, [0], lambda _: np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="Context"):
        generate_string(decoder, [], lambda _: np.zeros(1, dtype=np.float32))


def test_direct_trace_is_observational_and_cache_modes_match() -> None:
    decoder = make_decoder([b"a", b'"'])
    events: list[Any] = []
    source = [9]

    def scores(context: list[int]) -> NDArray[np.float32]:
        result = np.zeros(2, dtype=np.float32)
        result[0 if len(context) == 1 else 1] = 1
        return result

    fragment, ids, state = decoder.decode(
        source, scores, b"", lambda s, raw: s + raw,
        lambda s: {ord("a"), ord('"')} if s == b"" else {ord('"')},
        lambda s: s.endswith(b'"'), namespace="trace-test",
        trace=events.append,
    )
    assert fragment == b"a\""
    assert ids == [9, 0, 1]
    assert source == [9]
    assert state == fragment
    assert [event.complete for event in events] == [False, True]
    assert [event.byte_count for event in events] == [1, 2]
    assert all(event.max_steps == decoder.max_new_tokens for event in events)
    assert events[0].state == "b'a'"
    with pytest.raises(ValidationError):
        events[0].complete = True

    plain = decoder.decode(
        source, scores, b"", lambda s, raw: s + raw,
        lambda s: {ord("a"), ord('"')} if s == b"" else {ord('"')},
        lambda s: s.endswith(b'"'), namespace="trace-test",
    )
    assert plain == (fragment, ids, state)

    uncached = make_decoder([b"a", b'"'], cache=AllowedSetCache(capacity=0))
    cached = make_decoder([b"a", b'"'])
    fn1, _ = scripted_logits(uncached, [b"a", b'"'])
    fn2, _ = scripted_logits(cached, [b"a", b'"'])
    out1 = generate_string(uncached, [9], fn1)
    out2 = generate_string(cached, [9], fn2)
    assert (out1.value, out1.fragment, out1.context_ids) == (
        out2.value, out2.fragment, out2.context_ids
    )


def test_per_call_step_budget_and_observer_precedence() -> None:
    decoder = make_decoder([b"a", b'"'], max_new_tokens=3)
    observed: list[Any] = []
    explicit: list[Any] = []
    decoder.trace_observer = observed.append
    logits, _ = scripted_logits(decoder, [b"a", b'"'])

    result = generate_string(decoder, [0], logits)
    assert result.value == "a"
    assert len(observed) == 2

    override_decoder = make_decoder([b"a", b'"'], max_new_tokens=3)
    override_decoder.trace_observer = observed.append
    logits, _ = scripted_logits(override_decoder, [b"a", b'"'])
    override_decoder.decode(
        [0], logits, b"", lambda state, raw: state + raw,
        lambda state: {ord("a"), ord('"')} if state == b"" else {ord('"')},
        lambda state: state.endswith(b'"'), namespace="observer-precedence",
        trace=explicit.append,
    )
    assert len(explicit) == 2
    assert len(observed) == 2

    limited = make_decoder([b"a", b"b", b'"'], max_new_tokens=3)
    logits, _ = scripted_logits(limited, [b"a", b"b", b'"'])
    with pytest.raises(ValueError, match="token limit"):
        limited.decode(
            [0], logits, b"", lambda state, raw: state + raw,
            lambda state: {ord("a"), ord("b"), ord('"')},
            lambda state: state.endswith(b'"'), namespace="short-budget",
            max_steps=2,
        )

    global_cap = make_decoder([b"a", b"b", b'"'], max_new_tokens=2)
    logits, _ = scripted_logits(global_cap, [b"a", b"b", b'"'])
    with pytest.raises(ValueError, match="token limit"):
        global_cap.decode(
            [0], logits, b"", lambda state, raw: state + raw,
            lambda state: {ord("a"), ord("b"), ord('"')},
            lambda state: state.endswith(b'"'), namespace="global-budget",
            max_steps=10,
        )
    with pytest.raises(ValueError, match="at least one"):
        global_cap.decode(
            [0], logits, b"", lambda state, raw: state + raw,
            lambda state: {ord('"')}, lambda state: False,
            namespace="invalid-budget", max_steps=0,
        )


def test_seeded_random_logits_return_valid_or_bounded_failure() -> None:
    tokens = [b"a", b"b", b"\\", b"n", b'"', b"ab", b"a\""]
    rng = np.random.default_rng(42)
    decoder = make_decoder(tokens, max_new_tokens=8, max_value_bytes=8)

    def logits(_: list[int]) -> NDArray[np.float32]:
        return rng.normal(size=decoder.vocab.n_logits).astype(np.float32)

    try:
        result = generate_string(decoder, [0], logits)
    except ValueError:
        return
    assert isinstance(result.value, str)
    assert json.loads((b'"' + result.fragment).decode("utf-8")) == result.value


def test_finish_unique_matches_strict_choice_with_forced_suffix() -> None:
    """Append unique remaining bytes after the final scored choice."""
    raw_tokens = [b"fn", b"_add", b"_reverse", b'"']
    strict_decoder = make_decoder(raw_tokens)
    strict_logits, strict_calls = scripted_logits(
        strict_decoder, [b"fn", b"_add", b'"']
    )
    strict = generate_name(
        strict_decoder, [7], strict_logits, ["fn_add", "fn_reverse"]
    )

    optimized_decoder = make_decoder(raw_tokens)
    optimized_logits, optimized_calls = scripted_logits(
        optimized_decoder, [b"fn", b"_add"]
    )
    optimized = generate_name(
        optimized_decoder, [7], optimized_logits,
        ["fn_add", "fn_reverse"], finish_unique=True,
    )

    assert strict.value == optimized.value == "fn_add"
    assert strict.fragment == b'fn_add"'
    assert strict.forced_suffix == b""
    assert optimized.fragment == b"fn_add"
    assert optimized.forced_suffix == b'"'
    assert optimized.fragment + optimized.forced_suffix == strict.fragment
    assert strict_calls == [
        [7],
        [7, strict_decoder.vocab.id_to_bytes.index(b"fn")],
        [
            7,
            strict_decoder.vocab.id_to_bytes.index(b"fn"),
            strict_decoder.vocab.id_to_bytes.index(b"_add"),
        ],
    ]
    assert optimized_calls == [
        [7],
        [7, optimized_decoder.vocab.id_to_bytes.index(b"fn")],
    ]
    assert optimized.context_ids == [
        7,
        optimized_decoder.vocab.id_to_bytes.index(b"fn"),
        optimized_decoder.vocab.id_to_bytes.index(b"_add"),
    ]


def test_finish_unique_scores_ambiguous_terminal_prefix() -> None:
    """An ambiguous terminal prefix scores its quote against its branch."""
    decoder = make_decoder([b"a", b"b", b'"'])
    calls: list[list[int]] = []
    ids = {raw: index for index, raw in enumerate(decoder.vocab.id_to_bytes)
           if raw is not None}

    def logits(context: list[int]) -> NDArray[np.float32]:
        calls.append(list(context))
        scores = np.full(decoder.vocab.n_logits, -10.0, dtype=np.float32)
        scores[ids[b"a" if len(context) == 1 else b'"']] = 10.0
        return scores

    result = generate_name(
        decoder, [5], logits, ["a", "ab"], finish_unique=True
    )
    assert result.value == "a"
    assert result.fragment == b'a"'
    assert result.forced_suffix == b""
    assert len(calls) == 2
    assert calls[0] == [5]
    assert calls[1] == [5, ids[b"a"]]


def test_finish_unique_single_choice_still_uses_model_scores() -> None:
    """Even one remaining function requires at least one model call."""
    decoder = make_decoder([b"only", b'"'])
    logits, calls = scripted_logits(decoder, [b"only"])
    result = generate_name(
        decoder, [3], logits, ["only"], finish_unique=True
    )
    assert result.value == "only"
    assert result.fragment == b"only"
    assert result.forced_suffix == b'"'
    assert len(calls) == 1


def test_finish_unique_preserves_ambiguous_prefix_scores() -> None:
    """The chosen IDs stay the same while a name prefix remains ambiguous."""
    tokens = [b"a", b"b", b'"']
    decoder = make_decoder(tokens)
    logits, calls = scripted_logits(decoder, [b"a", b'"'])
    result = generate_name(
        decoder, [5], logits, ["a", "ab"], finish_unique=True
    )
    assert result.value == "a"
    assert result.fragment == b'a"'
    assert result.forced_suffix == b""
    assert calls == [[5], [5, tokens.index(b"a")]]
