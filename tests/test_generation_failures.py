"""Tests for typed generation failures and their immutable records."""

from typing import Any

import pytest
from pydantic import ValidationError

from src.decoding.failures import (
    RECOVERABLE_REASONS,
    DecodeFailure,
    FailureReason,
    FailureRecord,
    GenerationUsage,
    can_retry,
)


def make_usage(**overrides: object) -> GenerationUsage:
    """Build valid usage counters, allowing fields to be overridden."""
    values: dict[str, Any] = {
        "max_new_tokens": 8,
        "max_value_bytes": 64,
        "max_context_tokens": 128,
    }
    values.update(overrides)
    return GenerationUsage(**values)


def test_usage_defaults_and_validates_limits_and_counters() -> None:
    usage = make_usage()
    assert usage.generated_tokens == 0
    assert usage.generated_bytes == 0
    assert usage.context_tokens == 0
    assert usage.logits_calls == 0

    counter_fields = (
        "generated_tokens", "generated_bytes", "context_tokens", "logits_calls"
    )
    for field in counter_fields:
        with pytest.raises(ValidationError):
            make_usage(**{field: -1})
    for field in ("max_new_tokens", "max_value_bytes", "max_context_tokens"):
        with pytest.raises(ValidationError):
            make_usage(**{field: 0})
    with pytest.raises(ValidationError):
        make_usage(unexpected=True)
    with pytest.raises(ValidationError):
        make_usage(generated_tokens="2")


def test_usage_and_failure_record_are_frozen() -> None:
    usage = make_usage()
    record = FailureRecord(
        prompt_index=0,
        phase="parameter",
        argument_path=("items", 0),
        attempt=1,
        reason="invalid_value",
        message="invalid value",
        usage=usage,
    )
    with pytest.raises(ValidationError):
        usage.generated_tokens = 3
    with pytest.raises(ValidationError):
        record.attempt = 2


def test_failure_record_validates_and_serializes_to_json() -> None:
    usage = make_usage()
    record = FailureRecord(
        phase="selection",
        attempt=2,
        reason="token_limit",
        message="token limit reached",
        usage=usage,
        retry_scheduled=True,
    )
    dumped = record.model_dump(mode="json")
    assert dumped["prompt_index"] is None
    assert dumped["argument_path"] == []
    assert dumped["usage"]["max_new_tokens"] == 8
    assert record.model_dump_json()

    for invalid in (
        {"phase": "other", "attempt": 1, "reason": "token_limit"},
        {"phase": "selection", "attempt": 0, "reason": "token_limit"},
        {"phase": "selection", "attempt": 3, "reason": "token_limit"},
        {"phase": "selection", "attempt": 1, "reason": "unknown"},
        {
            "phase": "selection",
            "attempt": 1,
            "reason": "token_limit",
            "prompt_index": -1,
        },
    ):
        with pytest.raises(ValidationError):
            FailureRecord.model_validate(
                {"message": "oops", "usage": usage, **invalid}
            )


def test_decode_failure_preserves_value_error_compatibility() -> None:
    usage = make_usage()
    error = DecodeFailure("byte_limit", "output too large", usage)
    assert isinstance(error, ValueError)
    assert str(error) == "output too large"
    assert error.reason == "byte_limit"
    assert error.usage is usage


def test_retry_classification_uses_only_typed_reasons() -> None:
    usage = make_usage()
    expected = {
        "token_limit", "no_continuation", "nonfinite_number", "invalid_value"
    }
    assert RECOVERABLE_REASONS == frozenset(expected)
    for reason in RECOVERABLE_REASONS:
        assert can_retry(DecodeFailure(reason, "failed", usage))
    hard_failures: tuple[FailureReason, ...] = (
        "context_limit",
        "byte_limit",
        "invalid_logits",
    )
    for reason in hard_failures:
        assert not can_retry(DecodeFailure(reason, "failed", usage))
    assert not can_retry(ValueError("invalid_value"))
    assert not can_retry(OSError("token_limit"))
    assert not can_retry(RuntimeError("no_continuation"))
