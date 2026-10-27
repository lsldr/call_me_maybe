"""Numbered trace allocation must preserve existing runs."""

from datetime import date
from pathlib import Path

from src.diagnostics.trace_paths import allocate_trace


def test_trace_names_are_numbered_and_do_not_overwrite(tmp_path: Path) -> None:
    """Existing numbered traces, including older dates, remain untouched."""
    existing = tmp_path / "9_2026-01-01_previous.jsonl"
    existing.write_text("keep", encoding="utf-8")
    first = allocate_trace("baseline 11", tmp_path)
    second = allocate_trace("benchmark/ten", tmp_path)
    today = date.today().isoformat()
    assert first.name == f"10_{today}_baseline_11.jsonl"
    assert second.name == f"11_{today}_benchmark_ten.jsonl"
    assert first.read_bytes() == second.read_bytes() == b""
    assert existing.read_text(encoding="utf-8") == "keep"
