"""Tests for bounded model-free trace IO."""

import pytest
from pathlib import Path
from typing import Any
from pydantic import ValidationError

from src.diagnostics.trace_io import TraceReader, TraceWriter, convert_legacy
from src.diagnostics.trace_schema import StateSnapshot, TraceRecord


def token_data(**updates: Any) -> dict[str, Any]:
    data = {"step": 0, "token_id": 2, "raw_hex": "ff00", "allowed_count": 3,
            "byte_count": 2, "complete": True}
    data.update(updates)
    return data


def test_roundtrip_and_seek(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path=path).open()
    writer.emit("run_start", {"model": None})
    writer.emit("token", token_data())
    writer.finish({"status": "ok"})
    reader = TraceReader(path=path).open()
    assert len(reader) == 3
    assert reader[1].data["raw_hex"] == "ff00"
    assert [event.seq for event in reader] == [0, 1, 2]
    snapshot = StateSnapshot(kind="string", pending_hex="00", frames=("root",))
    assert snapshot.pending_hex == "00"


def test_caps_write_terminal_truncated_event(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path=path, max_events=2).open()
    assert writer.emit("run_start", {}) is not None
    assert writer.emit("attempt_end", {"status": "failed"}) is None
    terminal = writer.finish({"status": "failed"})
    assert terminal.data["truncated"] is True
    assert terminal.data["original_status"] == "failed"
    assert len(TraceReader(path=path).open()) == 2


def test_finish_reserves_compact_footer_and_keeps_status_on_oversized_end(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(
        path=path, max_record_bytes=256, max_file_bytes=300
    ).open()
    terminal = writer.finish({"status": "done", "detail": "x" * 500})
    assert terminal.data["truncated"] is True
    assert terminal.data["original_status"] == "done"

    too_small = TraceWriter(path=tmp_path / "small.jsonl", max_record_bytes=1)
    with pytest.raises(ValueError, match="terminal"):
        too_small.open()
    assert not too_small.path.exists()


def test_partial_line_policy_and_reader_limits(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_bytes(
        b'{"version":1,"kind":"run_start","seq":0,"data":{}}\n{"bad":')
    strict_reader = TraceReader(path=path)
    with pytest.raises(ValueError, match="partial"):
        strict_reader.open()
    partial_reader = TraceReader(path=path, allow_partial=True).open()
    assert partial_reader.partial
    assert len(partial_reader) == 1
    with pytest.raises(ValidationError):
        TraceReader(path=path, allow_partial=True, max_events=0).open()


def test_schema_rejects_bad_token_and_sequence_order(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        TraceRecord(kind="token", seq=0, data=token_data(raw_hex="nope"))
    path = tmp_path / "trace.jsonl"
    path.write_text(
        '{"version":1,"kind":"run_start","seq":1,"data":{}}\n'
        '{"version":1,"kind":"run_end","seq":0,"data":{}}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="increasing"):
        TraceReader(path=path).open()


def test_legacy_conversion_marks_provenance_and_case_id() -> None:
    events = convert_legacy([{
        "step": 4, "token_id": 8, "raw": "0a", "allowed_count": 1,
        "byte_count": 1, "complete": True,
        "prompt_index": "case-a", "state": "opaque legacy display",
    }])
    assert events[0].data["provenance"] == "legacy"
    assert events[0].data["raw_hex"] == "0a"
    assert events[0].data["case_id"] == "case-a"
    assert events[0].data["prompt_index"] is None
    assert events[0].data["legacy_state"] == "opaque legacy display"


def test_legacy_conversion_rejects_missing_required_token_fields() -> None:
    with pytest.raises(ValueError, match="required token fields"):
        convert_legacy([{"token_id": 1, "raw": "00"}])


@pytest.mark.parametrize("version", [True, 1.0, "1", 2])
def test_version_requires_the_exact_integer(version: object) -> None:
    """Boolean and float aliases cannot pass the trace format boundary."""
    with pytest.raises(ValidationError):
        TraceRecord.model_validate({
            "version": version, "kind": "run_start", "seq": 0,
        })


def test_open_twice_keeps_the_trace(tmp_path: Path) -> None:
    """Repeated opening rejects rather than leaking a handle or truncating."""
    writer = TraceWriter(path=tmp_path / "trace.jsonl").open()
    try:
        writer.emit("run_start", {})
        with pytest.raises(ValueError, match="already open"):
            writer.open()
        writer.finish({"status": "completed"})
    finally:
        writer.close()
    assert len(TraceReader(path=writer.path).open()) == 2


def test_live_observer_continues_after_trace_limit(tmp_path: Path) -> None:
    """A bounded recording does not stop the separate live event stream."""
    seen: list[TraceRecord] = []
    path = tmp_path / "limited.jsonl"
    writer = TraceWriter(path=path, max_events=2, observer=seen.append).open()
    writer.emit("run_start", {})
    writer.emit("token", token_data())
    writer.emit("token", token_data(step=1))
    writer.finish({"status": "completed"})
    assert [record.seq for record in seen] == [0, 1, 2, 3]
    assert [record.kind for record in seen] == [
        "run_start", "token", "token", "run_end",
    ]
    reader = TraceReader(path=path).open()
    assert reader.partial and len(reader) == 2
