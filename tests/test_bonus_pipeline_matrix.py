"""Compare scored generation with and without optional observation paths."""

from itertools import product
from pathlib import Path

import pytest

from src.config import RunSettings
from src.diagnostics.capture import attach_capture, start_capture
from src.diagnostics.trace_io import TraceReader, TraceWriter
from src.diagnostics.trace_schema import TraceRecord
from src.models.prompt import PromptItem
from test_recovery_decode_integration import _make_caller


def _raw_bytes(record: TraceRecord) -> bytes:
    """Decode raw_hex after narrowing the event's arbitrary JSON data."""
    raw_hex = record.data["raw_hex"]
    assert isinstance(raw_hex, str)
    return bytes.fromhex(raw_hex)


@pytest.mark.parametrize(
    ("optimize", "trace"), list(product((False, True), repeat=2))
)
def test_bonus_pipeline_matrix_keeps_scored_results_identical(
    tmp_path: Path, optimize: bool, trace: bool,
) -> None:
    """Optimization and live trace observation preserve generated values."""
    caller, contexts, _tokens, selection_scores = _make_caller(
        parameter_names=("first", "second"),
        value_targets={
            "first": (b"yes\"", b"yes\""),
            "second": (b"secret\"", b"ok\""),
        },
    )
    caller.decoder.optimize = optimize
    writer = None
    if trace:
        trace_path = tmp_path / f"optimized-{optimize}.jsonl"
        settings = RunSettings(profile="bonus", optimize=optimize)
        writer = TraceWriter(path=trace_path).open()
        start_capture(writer, settings)
        attach_capture(caller, writer)

    result = caller.call(PromptItem(prompt="request"), prompt_index=6)
    if writer is not None:
        writer.finish({"status": "completed"})

    assert result.name == "f"
    assert result.parameters == {"first": "yes", "second": "ok"}
    assert isinstance(result.parameters["first"], str)
    assert isinstance(result.parameters["second"], str)
    assert caller.functions[result.name].parameters.keys(
    ) == result.parameters.keys()
    assert selection_scores == [1]
    assert any("Return only the requested input value" in text for _,
               text in contexts)

    if trace:
        records = list(TraceReader(path=trace_path).open())
        run_start = records[0]
        assert run_start.kind == "run_start"
        assert run_start.data["provenance"] == "live_capture"
        settings_data = run_start.data["settings"]
        assert isinstance(settings_data, dict)
        assert settings_data["optimize"] is optimize
        attempt_starts = [
            event.data for event in records if event.kind == "attempt_start"
        ]
        assert attempt_starts[0] == {
            "prompt_index": 6, "phase": "selection", "path": [], "attempt": 1
        }
        retry = [item for item in attempt_starts if item["attempt"] == 2]
        assert retry == [{
            "prompt_index": 6, "phase": "parameter",
            "path": ["parameters", "second"], "attempt": 2,
        }]
        token_events = [event for event in records if event.kind == "token"]
        assert token_events
        assert all(event.data["typed_state_before"]
                   is not None for event in token_events)
        assert all(event.data["typed_state_after"]
                   is not None for event in token_events)
        assert all(isinstance(event.data["raw_hex"], str)
                   for event in token_events)


def test_recovery_trace_separates_failed_and_fresh_parameter_attempts(
    tmp_path: Path,
) -> None:
    """Retry token steps restart at zero and contain no failed prefix bytes."""
    caller, _contexts, _tokens, _selection = _make_caller(
        parameter_names=("first", "second"),
        value_targets={
            "first": (b"yes\"", b"yes\""),
            "second": (b"secret\"", b"ok\""),
        },
    )
    path = tmp_path / "recovery.jsonl"
    writer = TraceWriter(path=path).open()
    attach_capture(caller, writer)
    caller.call(PromptItem(prompt="request"), prompt_index=3)
    writer.finish({"status": "completed"})

    records = list(TraceReader(path=path).open())
    events = list(zip(range(len(records)), records))
    retry_start = next(
        position for position, record in events
        if record.kind == "attempt_start"
        and record.data.get("path") == ["parameters", "second"]
        and record.data.get("attempt") == 2
    )
    retry_end = next(
        position for position, record in events
        if position > retry_start and record.kind == "attempt_end"
    )
    retry_tokens = [
        record for position, record in events
        if retry_start < position < retry_end and record.kind == "token"
    ]
    assert retry_tokens
    assert [record.data["step"] for record in retry_tokens] == list(
        range(len(retry_tokens))
    )
    retry_bytes = b"".join(_raw_bytes(record) for record in retry_tokens)
    assert retry_bytes == b'ok"'
    assert b"secret" not in retry_bytes


def test_trace_soft_cap_does_not_change_model_choices(tmp_path: Path) -> None:
    """Trace truncation is observational even when only the footer fits."""
    baseline, _ctx, _events, _scores = _make_caller(
        parameter_names=("value",),
        value_targets={"value": (b"ok\"", b"ok\"")},
    )
    expected = baseline.call(PromptItem(prompt="request"))

    bounded, _ctx, _events, _scores = _make_caller(
        parameter_names=("value",),
        value_targets={"value": (b"ok\"", b"ok\"")},
    )
    path = tmp_path / "bounded.jsonl"
    writer = TraceWriter(path=path, max_events=2).open()
    start_capture(writer, RunSettings(profile="bonus"))
    attach_capture(bounded, writer)
    actual = bounded.call(PromptItem(prompt="request"))
    terminal = writer.finish({"status": "completed"})

    assert actual == expected
    assert terminal.data["truncated"] is True
    assert terminal.data["original_status"] == "completed"
    reader = TraceReader(path=path).open()
    records = list(reader)
    assert len(records) == 2
    assert records[-1].kind == "run_end"
    assert reader.partial
