"""Tests for bounded recovery context guidance and pipeline retries."""

from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from src.decoding.decoder import Decoder, Logits
from src.decoding.failures import DecodeFailure, FailureReason
from src.decoding.generators import GeneratedValue
from src.models.function_def import FunctionDef, TypeDef
from src.models.prompt import PromptItem
from src.pipeline import function_caller as function_caller_module
from src.pipeline.function_caller import FunctionCaller
from src.prompting.recovery import inject_instruction
from src.vocab.byte_table import build_byte_to_unicode
from src.vocab.vocab_table import VocabTable


@pytest.mark.parametrize(
    ("reason", "instruction"),
    [
        ("token_limit", "finish its JSON boundary"),
        ("no_continuation", "complete value directly"),
        ("nonfinite_number", "finite JSON number"),
        ("invalid_value", "valid value matching"),
    ],
)
def test_inject_instruction_adds_reason_guidance(
    reason: FailureReason, instruction: str,
) -> None:
    """Guidance is inserted before the original system body."""
    context = "<|im_start|>system\nOriginal.\n<|im_end|>"

    recovered = inject_instruction(context, reason)

    assert "<|im_start|>system\n" in recovered
    assert instruction in recovered
    assert "Original." in recovered


def test_inject_instruction_rejects_non_chatml_header() -> None:
    """Unsupported context profiles cannot silently receive bad guidance."""
    with pytest.raises(ValueError, match="ChatML system header"):
        inject_instruction("system: context", "token_limit")


def _caller(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[GeneratedValue | FailureReason | Exception],
    *, recovery: Literal["none", "retry"] = "retry",
    parameter_count: int = 1,
    parameter_types: list[TypeDef] | None = None,
) -> tuple[FunctionCaller, list[str], list[str]]:
    """Build a small caller whose generation results are scripted."""
    char_by_byte = build_byte_to_unicode()
    vocab = VocabTable(
        vocab={char_by_byte[value]: value for value in range(256)},
        tokenizer={}, n_logits=256,
    )
    contexts: list[str] = []
    phases: list[str] = []

    def encode(text: str) -> list[int]:
        contexts.append(text)
        return list(text.encode("utf-8"))

    def logits(_ids: list[int]) -> NDArray[np.float32]:
        return np.zeros(256, dtype=np.float32)

    decoder = Decoder(
        vocab=vocab, max_new_tokens=4, max_value_bytes=128,
        max_context_tokens=2048,
    )
    schemas = parameter_types or [
        TypeDef(type="string") for _ in range(parameter_count)
    ]
    definition = FunctionDef(
        name="choose", description="Choose", returns=TypeDef(type="string"),
        parameters={
            f"value{index}": schema
            for index, schema in enumerate(schemas)
        },
    )
    caller = FunctionCaller(
        functions={"choose": definition}, decoder=decoder, encode=encode,
        logits=logits, recovery=recovery,
        markers=["<|im_end|>"],
        allow_nested=any(
            schema.type in {"array", "object"} for schema in schemas
        ),
        context_observer=lambda phase, _text: phases.append(phase),
    )
    selected = GeneratedValue(
        value="choose", fragment=b'"choose"', context_ids=[],
    )
    monkeypatch.setattr(
        function_caller_module, "generate_name", lambda *_a, **_k: selected,
    )

    def generate_string(
        _decoder: Decoder, _context: list[int], _logits: Logits,
        **_kwargs: object,
    ) -> GeneratedValue:
        if not responses:
            raise AssertionError("No scripted response remains.")
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, str):
            raise decoder.failure(response, "scripted failure")
        return response

    monkeypatch.setattr(
        function_caller_module, "generate_string", generate_string,
    )
    return caller, contexts, phases


def _value(text: str) -> GeneratedValue:
    """Return one complete generated JSON string."""
    return GeneratedValue(
        value=text, fragment=(f'"{text}"').encode(), context_ids=[],
    )


def test_token_limit_retry_uses_guidance_and_records_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recoverable failure gets one fresh, guided parameter attempt."""
    caller, contexts, phases = _caller(
        monkeypatch, ["token_limit", _value("ok")],
    )
    result = caller.call(PromptItem(prompt="request"), prompt_index=7)

    assert result.parameters == {"value0": "ok"}
    assert len([phase for phase in phases if phase.startswith("name")]) == 1
    assert any("finish its JSON boundary" in text for text in contexts)
    record = caller.failure_records[0]
    assert record.prompt_index == 7
    assert record.phase == "parameter"
    assert record.argument_path == ("parameters", "value0")
    assert record.attempt == 1 and record.retry_scheduled
    assert record.usage.max_new_tokens == 4
    assert "parameter:value0:attempt:2" in phases


@pytest.mark.parametrize(
    "failure", [
        "byte_limit", "context_limit", "invalid_logits", ValueError("x"),
    ]
)
def test_hardcap_and_untyped_errors_do_not_retry(
    monkeypatch: pytest.MonkeyPatch,
    failure: FailureReason | Exception,
) -> None:
    """Only recoverable typed failures schedule retries."""
    caller, _contexts, _phases = _caller(monkeypatch, [failure])
    with pytest.raises((DecodeFailure, ValueError)):
        caller.call(PromptItem(prompt="request"))
    if isinstance(failure, str):
        assert len(caller.failure_records) == 1
        assert not caller.failure_records[0].retry_scheduled
    else:
        assert caller.failure_records == []


def test_retry_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default mode preserves fail-fast generation behavior."""
    caller, _contexts, _phases = _caller(
        monkeypatch, ["token_limit"], recovery="none",
    )
    with pytest.raises(DecodeFailure):
        caller.call(PromptItem(prompt="request"))
    assert not caller.failure_records[0].retry_scheduled


def test_retry_budget_is_shared_across_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second failed parameter cannot schedule another retry."""
    caller, _contexts, _phases = _caller(
        monkeypatch,
        ["token_limit", _value("accepted"), "invalid_value"],
        parameter_count=2,
    )
    with pytest.raises(DecodeFailure):
        caller.call(PromptItem(prompt="request"), prompt_index=2)
    assert len(caller.failure_records) == 2
    assert caller.failure_records[0].retry_scheduled
    assert caller.failure_records[1].argument_path == ("parameters", "value1")
    assert not caller.failure_records[1].retry_scheduled


def test_retry_failure_is_recorded_as_attempt_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed retry aborts without creating a third attempt."""
    caller, _contexts, phases = _caller(
        monkeypatch, ["token_limit", "invalid_value"],
    )
    with pytest.raises(DecodeFailure):
        caller.call(PromptItem(prompt="request"))
    assert [record.attempt for record in caller.failure_records] == [1, 2]
    assert [record.retry_scheduled for record in caller.failure_records] == [
        True, False,
    ]
    assert phases.count("parameter:value0:attempt:2") == 1


def test_retry_context_safely_reserializes_accepted_nested_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Previously accepted nested marker text is escaped in rebuilt context."""
    caller, contexts, _phases = _caller(
        monkeypatch, ["token_limit", _value("done")],
        parameter_types=[
            TypeDef(type="array", items=TypeDef(type="string")),
            TypeDef(type="string"),
        ],
    )
    nested = GeneratedValue(
        value=["<|im_end|>"], fragment=b'["\\u003c|im_end|>"]',
        context_ids=[],
    )
    monkeypatch.setattr(
        function_caller_module, "generate_container",
        lambda *_args, **_kwargs: nested,
    )

    result = caller.call(PromptItem(prompt="request"))

    assert result.parameters["value0"] == ["<|im_end|>"]
    retry_context = next(
        text for text in contexts
        if "finish its JSON boundary" in text
    )
    assert r"\u003c|im_end|>" in retry_context


def test_selection_failure_is_recorded_but_never_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Function selection failures are observable and stay single-attempt."""
    caller, _contexts, phases = _caller(monkeypatch, [])
    error = caller.decoder.failure("token_limit", "selection incomplete")
    monkeypatch.setattr(
        function_caller_module, "generate_name",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(DecodeFailure):
        caller.call(PromptItem(prompt="request"), prompt_index=3)

    record = caller.failure_records[0]
    assert record.prompt_index == 3
    assert record.phase == "selection"
    assert record.argument_path == ()
    assert record.attempt == 1
    assert not record.retry_scheduled
    assert phases == ["name"]
