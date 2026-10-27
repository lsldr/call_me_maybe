"""CLI recovery and diagnostics behavior without loading model weights."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from src import __main__ as cli
from src.decoding.failures import FailureRecord, GenerationUsage


def input_files(tmp_path: Path, prompts: object) -> tuple[Path, Path, Path]:
    """Write minimal valid function definitions and prompt input."""
    definitions = tmp_path / "functions.json"
    definitions.write_text(json.dumps([{
        "name": "choose", "description": "Choose", "parameters": {},
        "returns": {"type": "string"},
    }]))
    source = tmp_path / "input.json"
    source.write_text(json.dumps(prompts))
    return definitions, source, tmp_path / "output.json"


def invoke(
    monkeypatch: pytest.MonkeyPatch,
    paths: tuple[Path, Path, Path],
    *options: str,
) -> int:
    """Call main with paths and extra CLI options in the current process."""
    definitions, source, output = paths
    monkeypatch.setattr(sys, "argv", [
        "call-me-maybe", "--functions_definition", str(definitions),
        "--input", str(source), "--output", str(output), *options,
    ])
    return cli.main()


def retry_record() -> FailureRecord:
    """Build a representative failure that has a scheduled retry."""
    usage = GenerationUsage(
        generated_tokens=2, generated_bytes=3, context_tokens=8,
        logits_calls=2, max_new_tokens=4, max_value_bytes=64,
        max_context_tokens=128,
    )
    return FailureRecord(
        prompt_index=0, phase="parameter", argument_path=("parameters", "x"),
        attempt=1, reason="token_limit", message="incomplete value",
        usage=usage, retry_scheduled=True,
    )


class FakeCall:
    """Small stand-in for a serialized function call."""

    def model_dump(self) -> dict[str, object]:
        """Return a JSON-ready call record."""
        return {"name": "choose", "parameters": {}}


class FakeCaller:
    """Emit one recovery record, then either succeed or fail the batch."""

    def __init__(self, *, failure: BaseException | None = None) -> None:
        self.failure_observer: Any = None
        self.failure = failure
        self.observer_on_run: Any = None
        self.decoder = SimpleNamespace(trace_observer=None, max_value_bytes=64)
        self.attempt_observer: Any = None

    def run(self, prompts: list[object]) -> list[FakeCall]:
        """Simulate a retry event and return or fail the full batch."""
        assert len(prompts) == 1
        self.observer_on_run = self.failure_observer
        if self.failure_observer is not None:
            self.failure_observer(retry_record())
        if self.failure is not None:
            raise self.failure
        return [FakeCall()]


def install_factory(
    monkeypatch: pytest.MonkeyPatch, caller: FakeCaller,
) -> list[Any]:
    """Replace factory with a model-free caller and retain settings."""
    captured: list[Any] = []

    def factory(functions: object, settings: object) -> FakeCaller:
        captured.extend([functions, settings])
        return caller

    monkeypatch.setattr("src.pipeline.function_caller.create_caller", factory)
    return captured


def test_successful_retry_writes_batch_and_failure_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a recovered failure in diagnostics and publish the full batch."""
    paths = input_files(tmp_path, [{"prompt": "hello"}])
    diagnostics = tmp_path / "diagnostics.json"
    caller = FakeCaller()
    install_factory(monkeypatch, caller)

    status = invoke(
        monkeypatch, paths, "--profile", "bonus", "--recovery", "retry",
        "--diagnostics-output", str(diagnostics),
    )

    assert status == 0
    assert json.loads(paths[2].read_text()) == [
        {"name": "choose", "parameters": {}}
    ]
    report = json.loads(diagnostics.read_text())
    assert report["exit_status"] == 0
    assert report["error"] is None
    assert report["recovery"] == "retry"
    assert report["model_id"]
    assert report["revision"]
    assert report["snapshot_pinned"] is True
    assert report["failures"][0]["reason"] == "token_limit"
    assert caller.observer_on_run is not None


def test_exhausted_retry_preserves_output_and_writes_failure_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed batch preserves old output and writes diagnostics."""
    paths = input_files(tmp_path, [{"prompt": "hello"}])
    paths[2].write_text("output sentinel")
    diagnostics = tmp_path / "diagnostics.json"
    caller = FakeCaller(failure=ValueError("retry exhausted"))
    install_factory(monkeypatch, caller)

    status = invoke(
        monkeypatch, paths, "--profile", "bonus", "--recovery", "retry",
        "--diagnostics-output", str(diagnostics),
    )

    assert status == 1
    assert paths[2].read_text() == "output sentinel"
    report = json.loads(diagnostics.read_text())
    assert report["exit_status"] == 1
    assert report["error"] == "retry exhausted"
    assert len(report["failures"]) == 1


@pytest.mark.parametrize("prompts", ["malformed", []])
def test_bad_or_empty_input_does_not_create_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompts: object,
) -> None:
    """Malformed input fails before model creation; empty input skips it."""
    paths = input_files(tmp_path, [] if prompts == "malformed" else prompts)
    if prompts == "malformed":
        paths[1].write_text("{")
    diagnostics = tmp_path / "diagnostics.json"

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("caller must not be created")

    monkeypatch.setattr(
        "src.pipeline.function_caller.create_caller", forbidden_factory
    )
    status = invoke(
        monkeypatch, paths, "--profile", "bonus", "--recovery", "retry",
        "--diagnostics-output", str(diagnostics),
    )

    if prompts == "malformed":
        assert status == 1
        assert json.loads(diagnostics.read_text())["failures"] == []
    else:
        assert status == 0
        assert json.loads(paths[2].read_text()) == []
    assert diagnostics.exists()


@pytest.mark.parametrize("path_index", [0, 1, 2])
def test_diagnostics_collision_is_rejected_before_caller_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_index: int,
) -> None:
    """Diagnostics cannot replace any source or result path."""
    paths = input_files(tmp_path, [{"prompt": "hello"}])
    paths[2].write_text("sentinel")

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("path collision must be rejected first")

    monkeypatch.setattr(
        "src.pipeline.function_caller.create_caller", forbidden_factory
    )
    status = invoke(
        monkeypatch, paths, "--diagnostics-output", str(paths[path_index])
    )
    assert status == 1
    assert paths[2].read_text() == "sentinel"


def test_diagnostics_write_failure_returns_error_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit diagnostics destination that cannot be written fails CLI."""
    paths = input_files(tmp_path, [])
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.mkdir()
    status = invoke(
        monkeypatch, paths, "--diagnostics-output", str(diagnostics)
    )
    assert status == 1


def test_default_mandatory_mode_has_no_retry_or_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default CLI settings retain fail-fast behavior without a report file."""
    paths = input_files(tmp_path, [{"prompt": "hello"}])
    caller = FakeCaller()
    captured = install_factory(monkeypatch, caller)
    assert invoke(monkeypatch, paths) == 0
    settings = captured[1]
    assert settings.recovery == "none"
    assert settings.diagnostics_path is None
    assert caller.observer_on_run is None
    assert not (tmp_path / "diagnostics.json").exists()


def test_keyboard_interrupt_writes_diagnostics_and_returns_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupted runs preserve their failure records in diagnostics."""
    paths = input_files(tmp_path, [{"prompt": "hello"}])
    diagnostics = tmp_path / "diagnostics.json"
    install_factory(monkeypatch, FakeCaller(failure=KeyboardInterrupt()))
    status = invoke(
        monkeypatch, paths, "--profile", "bonus", "--recovery", "retry",
        "--diagnostics-output", str(diagnostics),
    )
    assert status == 130
    report = json.loads(diagnostics.read_text())
    assert report["exit_status"] == 130
    assert report["error"] == "Interrupted."
    assert len(report["failures"]) == 1
