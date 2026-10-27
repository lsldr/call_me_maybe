"""Resource and callback failure behavior for constrained decoding."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.decoding.decoder import Decoder
from src.decoding.failures import DecodeFailure
from src.decoding.generators import (
    generate_container,
    generate_number,
    generate_string,
)
from src.models.function_def import TypeDef
from src.schemas.nested import SchemaLimits
from src.vocab.byte_table import build_byte_to_unicode
from src.vocab.vocab_table import VocabTable


def make_decoder(tokens: list[bytes], **limits: Any) -> Decoder:
    """Create a decoder with exactly the supplied byte tokens."""
    alphabet = build_byte_to_unicode()
    vocab = {
        "".join(alphabet[byte] for byte in raw): token_id
        for token_id, raw in enumerate(tokens)
    }
    return Decoder(
        vocab=VocabTable(vocab=vocab, tokenizer={}, n_logits=len(tokens)),
        **limits,
    )


def scripted_logits(
    decoder: Decoder, token_ids: list[int]
) -> tuple[Callable[[list[int]], NDArray[np.float32]], list[list[int]]]:
    """Choose each scripted token in turn and retain each scored context."""
    calls: list[list[int]] = []

    def logits(context: list[int]) -> NDArray[np.float32]:
        calls.append(list(context))
        scores = np.full(decoder.vocab.n_logits, -1.0, dtype=np.float32)
        scores[token_ids[len(calls) - 1]] = 1.0
        return scores

    return logits, calls


def assert_failure(
    error: DecodeFailure,
    reason: str,
    *,
    tokens: int,
    byte_count: int,
    context: int,
    calls: int,
    max_new_tokens: int,
) -> None:
    """Check the stable reason and all recorded resource counters."""
    assert error.reason == reason
    assert error.usage.generated_tokens == tokens
    assert error.usage.generated_bytes == byte_count
    assert error.usage.context_tokens == context
    assert error.usage.logits_calls == calls
    assert error.usage.max_new_tokens == max_new_tokens


def test_token_limit_and_per_call_limit_are_reported() -> None:
    decoder = make_decoder([b"a"], max_new_tokens=4)
    logits, _ = scripted_logits(decoder, [0])
    with pytest.raises(DecodeFailure) as caught:
        generate_string(decoder, [9], logits, max_steps=1)
    assert_failure(
        caught.value, "token_limit", tokens=1, byte_count=1,
        context=2, calls=1, max_new_tokens=1,
    )


def test_byte_limit_counts_only_committed_output_and_preserves_trace() -> None:
    decoder = make_decoder([b"a", b'"'], max_value_bytes=1)
    logits, scored = scripted_logits(decoder, [0, 1])
    events: list[Any] = []
    decoder.trace_observer = events.append
    with pytest.raises(DecodeFailure) as caught:
        generate_string(decoder, [9], logits)
    assert_failure(
        caught.value, "byte_limit", tokens=1, byte_count=1,
        context=2, calls=2, max_new_tokens=decoder.max_new_tokens,
    )
    assert [event.raw for event in events] == [b"a"]
    assert scored == [[9], [9, 0]]


def test_context_limit_and_no_continuation_skip_logits() -> None:
    context_decoder = make_decoder([b'"'], max_context_tokens=1)
    context_calls: list[list[int]] = []

    def context_logits(ids: list[int]) -> NDArray[np.float32]:
        context_calls.append(ids)
        return np.zeros(1, dtype=np.float32)

    with pytest.raises(DecodeFailure) as context_error:
        generate_string(context_decoder, [4, 5], context_logits)
    assert_failure(
        context_error.value, "context_limit", tokens=0, byte_count=0,
        context=2, calls=0, max_new_tokens=context_decoder.max_new_tokens,
    )
    assert context_calls == []

    no_path = make_decoder([b"\\uD800"])
    logits, _ = scripted_logits(no_path, [0])
    with pytest.raises(DecodeFailure) as continuation_error:
        generate_string(no_path, [7], logits)
    assert_failure(
        continuation_error.value, "no_continuation", tokens=1,
        byte_count=len(b"\\uD800"), context=2, calls=1,
        max_new_tokens=no_path.max_new_tokens,
    )


@pytest.mark.parametrize("scores", [
    np.zeros(2, dtype=np.float32),
    np.array([np.nan], dtype=np.float32),
    np.array([np.inf], dtype=np.float32),
    np.array([-np.inf], dtype=np.float32),
])
def test_invalid_logits_report_reason_and_counters(
    scores: NDArray[np.float32],
) -> None:
    decoder = make_decoder([b'"'])
    calls: list[list[int]] = []

    def logits(context: list[int]) -> NDArray[np.float32]:
        calls.append(context)
        return scores

    with pytest.raises(DecodeFailure) as caught:
        generate_string(decoder, [3], logits)
    assert_failure(
        caught.value, "invalid_logits", tokens=0, byte_count=0,
        context=1, calls=1, max_new_tokens=decoder.max_new_tokens,
    )
    assert calls == [[3]]


def test_huge_finite_number_is_classified_as_nonfinite() -> None:
    huge = b"9" * 400
    decoder = make_decoder([huge + b"}"])
    logits, _ = scripted_logits(decoder, [0])
    with pytest.raises(DecodeFailure) as caught:
        generate_number(decoder, [2], logits)
    assert_failure(
        caught.value, "nonfinite_number", tokens=1,
        byte_count=len(huge) + 1, context=2, calls=1,
        max_new_tokens=decoder.max_new_tokens,
    )


@pytest.mark.parametrize(
    "exception", [OSError("callback"), RuntimeError("callback")]
)
def test_logits_callback_exceptions_propagate_unchanged(
    exception: Exception,
) -> None:
    decoder = make_decoder([b'"'])

    def logits(_: list[int]) -> NDArray[np.float32]:
        raise exception

    with pytest.raises(type(exception)) as caught:
        generate_string(decoder, [0], logits)
    assert caught.value is exception
    assert not isinstance(caught.value, DecodeFailure)


def test_nested_serialized_size_limit_is_byte_limit() -> None:
    token = b'{"x":1}'
    decoder = make_decoder([token])
    schema = TypeDef(
        type="object", properties={"x": TypeDef(type="integer")}
    )
    logits, _ = scripted_logits(decoder, [0])
    with pytest.raises(DecodeFailure) as caught:
        generate_container(
            decoder, [0], logits, schema, limits=SchemaLimits(max_bytes=2)
        )
    assert caught.value.reason == "byte_limit"
    assert caught.value.usage.generated_tokens == 1
    assert caught.value.usage.generated_bytes == len(token)
    assert caught.value.usage.logits_calls == 1
