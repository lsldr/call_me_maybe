"""Tests for safe offline terminal trace viewing."""

import builtins
import sys
from types import ModuleType
from pathlib import Path
from typing import Any

import pytest

from src.diagnostics.trace_io import TraceReader, TraceWriter
from src.diagnostics.trace_schema import TraceRecord
from src.visualization.plain import escape_terminal, render_record, render_view
from src.visualization.replay import ReplayState, interactive, state_at


def test_escape_terminal_removes_control_effects_and_newlines() -> None:
    """C0/C1 bytes are rendered literally, including terminal escapes."""
    assert escape_terminal("x\x1b[31m\n\t\x9b") == (
        "x\\x1b[31m\\x0a\\x09\\x9b"
    )


def test_renderer_displays_untrusted_values_safely() -> None:
    """Record values are escaped before plain text output."""
    record = TraceRecord(
        kind="run_start", seq=4, data={"model": "[red]\x1b]0;bad\x07"}
    )
    result = render_record(record)
    assert "[red]" in result
    assert "\\x1b" in result
    assert "\\x07" in result


def test_incremental_utf8_and_attempt_reset() -> None:
    """Split UTF-8 decodes across tokens and a new attempt clears it."""
    state = ReplayState()
    start = TraceRecord(kind="attempt_start", seq=0, data={
        "prompt_index": 0, "phase": "selection", "path": [], "attempt": 1
    })
    first = TraceRecord(kind="token", seq=1, data={
        "step": 0, "token_id": 3, "raw_hex": "e2", "allowed_count": 1,
        "byte_count": 1, "complete": False
    })
    second = TraceRecord(kind="token", seq=2, data={
        "step": 1, "token_id": 4, "raw_hex": "82ac", "allowed_count": 1,
        "byte_count": 2, "complete": True
    })
    state.apply(start, 0)
    state.apply(first, 1)
    assert state.pending_hex == "e2"
    state.apply(second, 2)
    assert state.text == "€"
    state.apply(start, 3)
    assert state.current_attempt_bytes == b""
    assert state.text == ""
    assert state.pending_hex == ""


def test_invalid_utf8_is_reported_as_hex_without_replacement() -> None:
    """Invalid token bytes are shown honestly without replacement glyphs."""
    state = ReplayState()
    token = TraceRecord(kind="token", seq=0, data={
        "step": 0, "token_id": 3, "raw_hex": "ff", "allowed_count": 1,
        "byte_count": 1, "complete": False
    })
    state.apply(token, 0)
    assert state.text == ""
    assert state.decode_error_hex == "ff"
    shown = render_view(token, state)
    assert "utf8_decode_error_hex=ff" in shown
    assert "�" not in shown


def test_indexed_seek_reconstructs_only_current_attempt(
    tmp_path: Path,
) -> None:
    """Seeking walks the indexed reader and discards prior attempts."""
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path=path).open()
    writer.emit("run_start", {"model": "sample-model"})
    writer.emit("attempt_start", {
        "prompt_index": 0, "phase": "selection", "path": [], "attempt": 1
    })
    writer.emit("token", {
        "step": 0, "token_id": 1, "raw_hex": "61", "allowed_count": 1,
        "byte_count": 1, "complete": True
    })
    writer.emit("attempt_start", {
        "prompt_index": 0, "phase": "parameter", "path": ["x"], "attempt": 2
    })
    writer.emit("token", {
        "step": 0, "token_id": 2, "raw_hex": "62", "allowed_count": 1,
        "byte_count": 1, "complete": True
    })
    writer.finish()
    reader = TraceReader(path=path).open()
    state = state_at(reader, 4)
    assert state.current_attempt_bytes == b"b"
    assert state.text == "b"
    assert state.model == "sample-model"
    assert state.attempt["attempt"] == 2
    assert any(frame.get("kind") == "attempt" for frame in state.frames)
    assert "current_model=sample-model" in render_view(
        reader[4], state
    )


def test_navigation_clamps_commands_and_eof_quits(tmp_path: Path) -> None:
    """Line navigation handles bounds and treats EOF as quit."""
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path=path).open()
    writer.emit("run_start", {"model": "m"})
    writer.finish()
    reader = TraceReader(path=path).open()
    commands = iter(["prev", "goto 999", "next", "quit"])
    shown: list[str] = []
    interactive(
        reader, input_fn=lambda _: next(commands), output_fn=shown.append
    )
    assert len(shown) == 4
    with pytest.raises(ValueError, match="positive and finite"):
        interactive(reader, speed=0)
    interactive(
        reader,
        input_fn=lambda _: (_ for _ in ()).throw(EOFError()),
    )


def test_optional_rich_falls_back_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional renderer returns plain output when Rich cannot import."""
    from src.visualization.rich_view import render_rich

    original_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("rich"):
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    record = TraceRecord(kind="run_start", seq=0, data={})
    assert render_rich(record) == render_record(record)


def test_rich_renderer_builds_literal_known_field_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Table rows expose known fields while escaping trace control bytes."""
    from src.visualization.rich_view import render_rich

    class FakeText:
        def __init__(self, text: str, **_: object) -> None:
            self.text = text

        def __str__(self) -> str:
            return self.text

    class FakeTable:
        rows: list[tuple[str, str]]

        def __init__(self, **_: object) -> None:
            self.rows = []

        def add_column(self, *_: object, **__: object) -> None:
            pass

        def add_row(self, first: FakeText, second: FakeText) -> None:
            self.rows.append((first.text, second.text))

    class FakeCapture:
        def __init__(self, console: "FakeConsole") -> None:
            self.console = console

        def __enter__(self) -> "FakeCapture":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get(self) -> str:
            assert self.console.table is not None
            return "\n".join(
                f"{key} | {value}"
                for key, value in self.console.table.rows
            )

    class FakeConsole:
        table: FakeTable | None = None

        def __init__(self, **_: object) -> None:
            self.table = None

        def capture(self) -> FakeCapture:
            return FakeCapture(self)

        def print(self, table: FakeTable, **_: object) -> None:
            self.table = table

    modules = {
        "rich.console": ModuleType("rich.console"),
        "rich.table": ModuleType("rich.table"),
        "rich.text": ModuleType("rich.text"),
    }
    modules["rich.console"].Console = FakeConsole  # type: ignore[attr-defined]
    modules["rich.table"].Table = FakeTable  # type: ignore[attr-defined]
    modules["rich.text"].Text = FakeText  # type: ignore[attr-defined]
    original_import = builtins.__import__

    def fake_import(
        name: str, globals_: Any = None, locals_: Any = None,
        fromlist: tuple[str, ...] = (), level: int = 0,
    ) -> Any:
        if name in modules:
            return modules[name]
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    record = TraceRecord(kind="token", seq=2, data={
        "step": 3, "token_id": 17, "raw_hex": "611b", "allowed_count": 4,
        "byte_count": 2, "complete": True,
        "typed_state_before": {"kind": "string", "phase": "NORMAL"},
        "typed_state_after": {"kind": "string", "phase": "DONE"},
        "validation": "passed", "budget": {"tokens": 8, "bytes": 40},
    })
    state = ReplayState(model="model-id", text="a\x1b[31m")

    rendered = render_rich(record, state)

    for field in (
        "run", "model", "phase", "path", "attempt", "step", "token ID",
        "raw hex", "decoded text", "pending UTF-8", "allowed count",
        "typed state before", "typed state after", "frames",
        "token/byte budget", "token bytes", "final validation", "status",
    ):
        assert field in rendered
    assert "a\\x1b[31m" in rendered
    assert "unavailable" in rendered
    assert "\x1b" not in rendered


def test_rich_plain_mode_is_exact_plain_fallback() -> None:
    """Explicit no-color rendering always matches the plain viewer."""
    from src.visualization.rich_view import render_rich

    record = TraceRecord(kind="run_start", seq=0, data={"model": "m"})
    state = ReplayState(model="m")
    assert render_rich(record, state, color=False) == render_view(
        record, state,
    )


def test_viewer_handles_missing_and_invalid_trace_files(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Unreadable paths and malformed records return a clean status code."""
    from src.visualization import __main__ as viewer

    missing = tmp_path / "missing.jsonl"
    monkeypatch.setattr(sys, "argv", ["viewer", str(missing)])
    assert viewer.main() == 1
    assert "Cannot read trace:" in capsys.readouterr().err

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("not json\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["viewer", str(invalid)])
    assert viewer.main() == 1
    assert "Cannot read trace:" in capsys.readouterr().err


def test_viewer_partial_flag_warns_and_reads_complete_records(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The explicit partial mode shows complete records and warns on stderr."""
    from src.visualization import __main__ as viewer

    path = tmp_path / "partial.jsonl"
    writer = TraceWriter(path=path).open()
    writer.emit("run_start", {"model": "fixture"})
    writer.finish()
    with path.open("ab") as stream:
        stream.write(b'{"unfinished":')

    monkeypatch.setattr(sys, "argv", [
        "viewer", str(path), "--plain", "--allow-partial",
    ])
    assert viewer.main() == 0

    captured = capsys.readouterr()
    assert "partial or truncated" in captured.err
    assert "model=fixture" in captured.out


def test_viewer_keyboard_interrupt_returns_130(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Interrupting record rendering is reported as a handled CLI exit."""
    from src.visualization import __main__ as viewer

    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path=path).open()
    writer.emit("run_start", {})
    writer.finish()
    monkeypatch.setattr(sys, "argv", ["viewer", str(path), "--plain"])
    monkeypatch.setattr(
        viewer, "render_view",
        lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    assert viewer.main() == 130
    assert "Interrupted." in capsys.readouterr().err


def test_viewer_menu_orders_numeric_prefix_and_opens_selection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Menu order uses numeric prefixes and opens the selected trace."""
    from src.visualization import __main__ as viewer

    for name, model in (("2_second.jsonl", "two"),
                        ("10_tenth.jsonl", "ten")):
        writer = TraceWriter(path=tmp_path / name).open()
        writer.emit("run_start", {"model": model})
        writer.finish()
    monkeypatch.setattr(sys, "argv", [
        "viewer", "--trace-dir", str(tmp_path), "--plain",
    ])
    monkeypatch.setattr(builtins, "input", lambda _: "1")

    assert viewer.main() == 0
    captured = capsys.readouterr()
    assert captured.out.index("1. 10_tenth.jsonl") < captured.out.index(
        "2. 2_second.jsonl"
    )
    assert "model=ten" in captured.out
    assert "model=two" not in captured.out


def test_viewer_menu_reprompts_on_invalid_and_out_of_range(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Malformed and out-of-range choices do not select any file."""
    from src.visualization import __main__ as viewer

    path = tmp_path / "1_trace.jsonl"
    writer = TraceWriter(path=path).open()
    writer.emit("run_start", {"model": "selected"})
    writer.finish()
    answers = iter(["nonsense", "0", "2", "1"])
    monkeypatch.setattr(sys, "argv", [
        "viewer", "--trace-dir", str(tmp_path), "--plain",
    ])
    monkeypatch.setattr(builtins, "input", lambda _: next(answers))

    assert viewer.main() == 0
    captured = capsys.readouterr()
    assert "Enter a listed number or q to quit." in captured.out
    assert "That trace number is out of range." in captured.out
    assert "model=selected" in captured.out


def test_viewer_menu_empty_directory_and_eof(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """An empty menu is an error while EOF exits a populated menu cleanly."""
    from src.visualization import __main__ as viewer

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "viewer", "--trace-dir", str(empty),
    ])
    assert viewer.main() == 1
    assert "No trace files found" in capsys.readouterr().err

    path = tmp_path / "3_trace.jsonl"
    writer = TraceWriter(path=path).open()
    writer.emit("run_start", {})
    writer.finish()
    monkeypatch.setattr(sys, "argv", [
        "viewer", "--trace-dir", str(tmp_path),
    ])
    monkeypatch.setattr(
        builtins, "input", lambda _: (_ for _ in ()).throw(EOFError())
    )
    assert viewer.main() == 0


def test_imports_stay_offline_and_do_not_load_model_packages() -> None:
    """The viewer import graph excludes SDK, tokenizer, and GPU modules."""
    import subprocess
    import sys

    result = subprocess.run([
        sys.executable, "-c",
        "import src.visualization.__main__; import sys; "
        "assert not any(n == 'llm_sdk' or "
        "n.startswith(('torch', 'transformers', 'tokenizers')) "
        "for n in sys.modules)",
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
