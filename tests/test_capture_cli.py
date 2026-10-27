"""Exercise capture failures and path validation through the production CLI."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.config import RunSettings
from src.decoding.tracer import TraceEvent
from src.diagnostics.trace_io import TraceReader, TraceWriter
from src.diagnostics.trace_schema import StateSnapshot
from src.models.prompt import PromptItem
from test_recovery_decode_integration import _make_caller


@pytest.mark.parametrize(
    ("field", "protected"),
    [
        ("diagnostics_path", "output_path"),
        ("diagnostics_path", "input_path"),
        ("diagnostics_path", "functions_path"),
        ("performance_path", "output_path"),
        ("performance_path", "input_path"),
        ("performance_path", "functions_path"),
        ("trace_path", "output_path"),
        ("trace_path", "input_path"),
        ("trace_path", "functions_path"),
    ],
)
def test_artifact_paths_cannot_alias_input_or_batch_output(
    tmp_path: Path, field: str, protected: str,
) -> None:
    """Run settings reject artifact paths that could replace inputs/results."""
    values: dict[str, Any] = {
        "profile": "bonus",
        "output_path": tmp_path / "out.json",
        "input_path": tmp_path / "in.json",
        "functions_path": tmp_path / "functions.json",
        "diagnostics_path": tmp_path / "diag.json",
        "performance_path": tmp_path / "perf.json",
        "trace_path": tmp_path / "trace.jsonl",
    }
    values[field] = values[protected]
    with pytest.raises(ValueError, match="separate output files"):
        RunSettings(**values)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("diagnostics_path", "performance_path"),
        ("diagnostics_path", "trace_path"),
        ("performance_path", "trace_path"),
    ],
)
def test_diagnostic_artifacts_cannot_alias_each_other(
    tmp_path: Path, left: str, right: str,
) -> None:
    """Every sidecar needs a distinct destination before inference starts."""
    values: dict[str, Any] = {
        "profile": "bonus",
        "diagnostics_path": tmp_path / "diag.json",
        "performance_path": tmp_path / "perf.json",
        "trace_path": tmp_path / "trace.jsonl",
    }
    values[right] = values[left]
    with pytest.raises(ValueError, match="separate output files"):
        RunSettings(**values)


def test_fatal_trace_observer_io_error_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An observer OSError escapes the generation attempt without recovery."""
    caller, _contexts, _trace, _selection = _make_caller(
        parameter_names=("value",),
        value_targets={"value": (b"ok\"", b"retry\"")},
    )
    path = tmp_path / "fatal.jsonl"
    writer = TraceWriter(path=path).open()
    from src.diagnostics.capture import attach_capture

    attach_capture(caller, writer)
    original_write = TraceWriter._write
    writes = 0

    def fail_on_first_token(self: TraceWriter, encoded: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes >= 2:
            raise OSError("trace disk failure")
        original_write(self, encoded)

    monkeypatch.setattr(TraceWriter, "_write", fail_on_first_token)
    with pytest.raises(OSError, match="trace disk failure"):
        caller.call(PromptItem(prompt="request"))
    assert writes == 3  # Attempt start, failed token, and failed attempt_end.
    writer.close()
    reader = TraceReader(path=path).open()
    assert reader.partial
    starts = [event for event in reader if event.kind == "attempt_start"]
    assert len(starts) == 1
    assert starts[0].data["attempt"] == 1


class _FakeDecoder:
    """Minimal decoder surface used by the CLI orchestration test."""

    def __init__(self) -> None:
        self.trace_observer: Any = None
        self.max_value_bytes = 128


class _FakeCaller:
    """Duck-typed caller that emits exactly one committed token."""

    def __init__(self) -> None:
        self.decoder = _FakeDecoder()
        self.attempt_observer: Any = None
        self.failure_observer: Any = None
        self.performance = None

    def run(self, prompts: list[PromptItem]) -> list[Any]:
        assert prompts
        if self.attempt_observer is not None:
            self.attempt_observer("attempt_start", {
                "prompt_index": 0, "phase": "selection", "path": [],
                "attempt": 1,
            })
        event = TraceEvent(
            step=0, token_id=65, raw=b"A", allowed_count=1, complete=False,
            state="display only", byte_count=1, max_steps=4,
            before=StateSnapshot(kind="string"),
            after=StateSnapshot(kind="string"),
        )
        self.decoder.trace_observer(event)
        return [SimpleNamespace(model_dump=lambda: {"name": "x"})]


def test_cli_keeps_old_batch_output_when_trace_observer_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trace IO failure aborts publication and preserves the previous file."""
    from src import __main__ as cli

    output = tmp_path / "calls.json"
    output.write_text("old complete batch", encoding="utf-8")
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "call-me-maybe", "--profile", "bonus", "--output", str(output),
        "--input", "data/input/function_calling_tests.json",
        "--functions_definition", "data/input/functions_definition.json",
        "--trace-output", str(trace_path),
    ])
    monkeypatch.setattr(
        "src.pipeline.function_caller.create_caller",
        lambda _functions, _settings: _FakeCaller(),
    )
    original_write = TraceWriter._write
    writes = 0

    def fail_on_token(self: TraceWriter, encoded: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("trace disk failure")
        original_write(self, encoded)

    monkeypatch.setattr(TraceWriter, "_write", fail_on_token)
    assert cli.main() == 1
    assert writes == 4  # run_start, attempt_start, token, then run_end
    assert output.read_text(encoding="utf-8") == "old complete batch"


def test_invalid_trace_destination_fails_before_caller_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trace preflight happens before any caller can load a model."""
    from src import __main__ as cli

    output = tmp_path / "calls.json"
    output.write_text("previous", encoding="utf-8")
    missing_trace = tmp_path / "missing" / "trace.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "call-me-maybe", "--profile", "bonus", "--output", str(output),
        "--input", "data/input/function_calling_tests.json",
        "--functions_definition", "data/input/functions_definition.json",
        "--trace-output", str(missing_trace),
    ])

    def forbidden_create(_functions: Any, _settings: Any) -> Any:
        raise AssertionError("caller creation must follow trace preflight")

    monkeypatch.setattr(
        "src.pipeline.function_caller.create_caller", forbidden_create
    )
    assert cli.main() == 1
    assert output.read_text(encoding="utf-8") == "previous"


def test_live_cli_displays_tokens_before_the_caller_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live observer sees committed tokens before generation completes."""
    from src import __main__ as cli
    from src.visualization.live import LiveRenderer

    seen: list[str] = []
    observe = LiveRenderer.observe

    def track(renderer: LiveRenderer, record: Any) -> None:
        seen.append(record.kind)
        observe(renderer, record)

    class Caller(_FakeCaller):
        def run(self, prompts: list[PromptItem]) -> list[Any]:
            calls = super().run(prompts)
            assert "token" in seen
            return calls

    monkeypatch.setattr(LiveRenderer, "observe", track)
    monkeypatch.setattr(
        "src.pipeline.function_caller.create_caller", lambda *_: Caller(),
    )
    path = tmp_path / "run.jsonl"
    settings = RunSettings(
        visual=True, trace_path=path, output_path=tmp_path / "calls.json",
    )
    cli._run(settings, [])
    assert seen[0] == "run_start" and seen[-1] == "run_end"
    assert "attempt_start" in seen and "token" in seen
    assert not TraceReader(path=path).open().partial
