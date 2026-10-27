"""Exercise bounded recovery through the real byte-level decoder."""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.decoding.decoder import Decoder
from src.decoding.failures import DecodeFailure
from src.decoding.tracer import TraceEvent
from src.models.function_def import FunctionDef, TypeDef
from src.models.prompt import PromptItem
from src.pipeline.function_caller import FunctionCaller
from src.vocab.byte_table import build_byte_to_unicode
from src.vocab.vocab_table import VocabTable


def _make_caller(
    *, parameter_names: tuple[str, ...],
    value_targets: dict[str, tuple[bytes, bytes]],
) -> tuple[FunctionCaller, list[tuple[str, str]], list[TraceEvent], list[int]]:
    """Build a caller with byte logits targeting contexts by suffix."""
    char_by_byte = build_byte_to_unicode()
    vocab = VocabTable(
        vocab={char_by_byte[value]: value for value in range(256)},
        tokenizer={}, n_logits=256,
    )
    observed: list[tuple[str, str]] = []
    trace: list[TraceEvent] = []
    selection_calls: list[int] = []
    decoder = Decoder(
        vocab=vocab, max_new_tokens=4, max_value_bytes=128,
        max_context_tokens=4096,
        trace_observer=trace.append,
    )
    definition = FunctionDef(
        name="f", description="Choose", returns=TypeDef(type="string"),
        parameters={name: TypeDef(type="string") for name in parameter_names},
    )

    def encode(text: str) -> list[int]:
        observed.append(("encoded", text))
        return list(text.encode("utf-8"))

    def logits(ids: list[int]) -> NDArray[np.float32]:
        raw = bytes(ids)
        candidates = [(b'{"name": "', b'f"')]
        for key, (initial, retry) in value_targets.items():
            target = (
                retry if b"Return only the requested input value" in raw
                else initial
            )
            candidates.append((f'"{key}": "'.encode(), target))
        chosen: int | None = None
        for prefix, target in candidates:
            for consumed in range(len(target)):
                if raw.endswith(prefix + target[:consumed]):
                    chosen = target[consumed]
                    if prefix == b'{"name": "':
                        selection_calls.append(1)
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise AssertionError(f"No byte continuation for {raw[-120:]!r}")
        scores = np.full(256, -1000.0, dtype=np.float32)
        scores[chosen] = 1000.0
        return scores

    caller = FunctionCaller(
        functions={"f": definition}, decoder=decoder, encode=encode,
        logits=logits, recovery="retry", markers=[],
        context_observer=lambda phase, text: observed.append((phase, text)),
    )
    return caller, observed, trace, selection_calls


def test_real_decoder_discards_partial_string_and_retries_at_same_cap(
) -> None:
    """Partial first output is not prefixed to the guided second attempt."""
    caller, observed, trace, selection_calls = _make_caller(
        parameter_names=("first", "second"),
        value_targets={"first": (b"yes\"", b"yes\""),
                       "second": (b"secret\"", b"ok\"")},
    )

    result = caller.call(PromptItem(prompt="request"), prompt_index=5)

    assert result.parameters == {"first": "yes", "second": "ok"}
    assert len(selection_calls) == 1  # The unique name is scored once.
    failure = caller.failure_records[0]
    assert failure.prompt_index == 5
    assert failure.phase == "parameter"
    assert failure.argument_path == ("parameters", "second")
    assert failure.reason == "token_limit"
    assert failure.usage.generated_tokens == 4
    assert failure.usage.generated_bytes == 4
    assert failure.usage.logits_calls == 4
    assert failure.usage.max_new_tokens == caller.decoder.max_new_tokens == 4
    retry_context = next(
        text for phase, text in observed
        if phase == "parameter:second:attempt:2"
    )
    assert "secret" not in retry_context
    assert '"first":"yes"' in retry_context
    assert b"secr" in b"".join(event.raw for event in trace)


def test_selection_observer_fires_on_failure_without_changing_generation(
) -> None:
    """Selection observation reports the validated name without scoring it."""
    targets = {
        "first": (b"yes\"", b"yes\""),
        "second": (b"secret\"", b"ok\""),
    }
    observed_caller, observed_contexts, observed_trace, _ = _make_caller(
        parameter_names=("first", "second"), value_targets=targets,
    )
    selected: list[str] = []
    observed_caller.selection_observer = selected.append

    result = observed_caller.call(PromptItem(prompt="request"))

    baseline_caller, baseline_contexts, baseline_trace, _ = _make_caller(
        parameter_names=("first", "second"), value_targets=targets,
    )
    baseline_result = baseline_caller.call(PromptItem(prompt="request"))

    assert selected == ["f"]
    assert result == baseline_result
    assert observed_contexts == baseline_contexts
    assert [event.token_id for event in observed_trace] == [
        event.token_id for event in baseline_trace
    ]


def test_smollm2_context_guard_rejects_experimental_style_before_scoring(
) -> None:
    """Unsupported profile/style combinations fail before model selection."""
    caller, _observed, _trace, selection_calls = _make_caller(
        parameter_names=("value",), value_targets={"value": (b"x\"", b"x\"")},
    )
    caller.context_family = "smollm2"

    with pytest.raises(ValueError, match="style"):
        caller.style = "focused"  # type: ignore[assignment]

    assert selection_calls == []


def test_real_decoder_cannot_retry_a_later_parameter_after_budget_used(
) -> None:
    """A successful retry consumes the sole per-prompt retry opportunity."""
    caller, observed, _trace, selection_calls = _make_caller(
        parameter_names=("first", "second"),
        value_targets={"first": (b"secret\"", b"ok\""),
                       "second": (b"later\"", b"yes\"")},
    )

    selected: list[str] = []
    caller.selection_observer = selected.append
    try:
        caller.call(PromptItem(prompt="request"), prompt_index=2)
    except DecodeFailure:
        pass
    else:
        raise AssertionError("The over-budget second parameter should fail.")

    assert len(selection_calls) == 1
    assert selected == ["f"]
    assert [record.retry_scheduled for record in caller.failure_records] == [
        True, False,
    ]
    assert [record.argument_path for record in caller.failure_records] == [
        ("parameters", "first"), ("parameters", "second"),
    ]
    assert not any(
        phase == "parameter:second:attempt:2" for phase, _text in observed
    )
