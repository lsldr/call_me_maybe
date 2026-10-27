"""Tests for incremental live trace rendering."""

import builtins
from io import StringIO
from typing import Any

import pytest

from src.diagnostics.trace_schema import TraceRecord
from src.visualization.live import LiveRenderer


def record(kind: str, seq: int, data: dict[str, object]) -> TraceRecord:
    return TraceRecord(kind=kind, seq=seq, data=data)  # type: ignore[arg-type]


def test_plain_output_is_incremental_and_shows_token_state() -> None:
    stream = StringIO()
    renderer = LiveRenderer(stream=stream, terminal=False)
    with renderer:
        renderer.observe(record("run_start", 0, {"model": "qwen"}))
        renderer.observe(record("attempt_start", 1, {
            "prompt_index": 0, "phase": "parameter", "path": ["name"],
            "attempt": 1,
        }))
        renderer.observe(record("token", 2, {
            "step": 0, "token_id": 42, "raw_hex": "6869",
            "allowed_count": 3, "byte_count": 2, "complete": False,
            "typed_state_before": {"phase": "string"},
            "typed_state_after": {"phase": "string", "bytes": 2},
        }))
        before_close = stream.getvalue()
        assert "kind=token" in before_close
        assert "decoded=hi" in before_close
        assert "allowed_count=3" in before_close
        assert "typed_state_after" in before_close
    assert stream.getvalue().count("kind=") == 3


def test_terminal_controls_are_escaped() -> None:
    stream = StringIO()
    renderer = LiveRenderer(stream=stream, terminal=False)
    renderer.observe(record("run_start", 0, {"model": "qwen\x1b[31m"}))
    assert "qwen\\x1b[31m" in stream.getvalue()
    assert "\x1b" not in stream.getvalue()


def test_retry_attempt_resets_decoded_state() -> None:
    stream = StringIO()
    renderer = LiveRenderer(stream=stream, terminal=False)
    renderer.observe(record("attempt_start", 0, {
        "prompt_index": 0, "phase": "parameter", "path": [],
        "attempt": 1,
    }))
    renderer.observe(record("token", 1, {
        "step": 0, "token_id": 1, "raw_hex": "6f6c64",
        "allowed_count": 1, "byte_count": 3, "complete": False,
    }))
    renderer.observe(record("attempt_start", 2, {
        "prompt_index": 0, "phase": "parameter", "path": [],
        "attempt": 2,
    }))
    renderer.observe(record("token", 3, {
        "step": 0, "token_id": 2, "raw_hex": "6e6577",
        "allowed_count": 1, "byte_count": 3, "complete": False,
    }))
    retry_line = next(
        line for line in stream.getvalue().splitlines()
        if "seq=2 | kind=attempt_start" in line
    )
    assert "decoded=old" not in retry_line
    assert "decoded=new" in stream.getvalue()


def test_missing_rich_uses_incremental_plain_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def import_without_rich(name: str, *args: Any,
                            **kwargs: Any) -> Any:
        if name.startswith("rich"):
            raise ImportError("Rich is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_rich)
    stream = StringIO()
    renderer = LiveRenderer(stream=stream, terminal=True).open()
    renderer.observe(record("run_start", 0, {"model": "qwen"}))
    assert "kind=run_start" in stream.getvalue()
    renderer.close()


def test_rich_terminal_refreshes_each_event_with_literal_content() -> None:
    stream = StringIO()
    renderer = LiveRenderer(stream=stream, terminal=True)
    renderer.open()
    renderer.observe(record("run_start", 0, {
        "model": "qwen\x1b]0;OWNED\x07",
    }))
    renderer.observe(record("attempt_start", 1, {
        "prompt_index": 0, "phase": "parameter", "path": ["name"],
        "attempt": 1,
    }))
    renderer.observe(record("token", 2, {
        "step": 0, "token_id": 42, "raw_hex": "6869",
        "allowed_count": 3, "byte_count": 2, "complete": False,
    }))
    renderer.close()
    output = stream.getvalue()
    assert "decoded text" in output
    assert "hi" in output
    assert "qwen\\x1b]0;OWNED\\x07" in output


def test_module_does_not_import_model_sdk() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable, "-c",
            "import src.visualization.live; import sys; "
            "assert not any(n == 'llm_sdk' or "
            "n.startswith(('torch', 'transformers', 'tokenizers')) "
            "for n in sys.modules)",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
