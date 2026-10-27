"""Optional live visualization for model-free pytest runs."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pathlib import Path

from src.decoding.decoder import Decoder
from src.decoding.tracer import TraceEvent
from src.diagnostics.trace_io import TraceWriter
from src.diagnostics.trace_schema import TraceRecord
from src.diagnostics.trace_paths import allocate_trace
from src.visualization.live import LiveRenderer

pytest_plugins = ("pytester",)


@pytest.fixture(autouse=True)
def _isolate_application_traces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Keep fake application runs from polluting the real trace directory."""
    from src import __main__ as cli

    monkeypatch.setattr(
        cli, "allocate_trace",
        lambda label: allocate_trace(label, tmp_path / "application-traces"),
    )
    from src.diagnostics import trace_paths

    monkeypatch.setattr(
        trace_paths, "allocate_trace",
        lambda label: allocate_trace(label, tmp_path / "benchmark-traces"),
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the opt-in visual test-run switch."""
    parser.addoption(
        "--visual", action="store_true", default=False,
        help="stream constrained generation progress during pytest",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register visualization state only when requested."""
    visual = bool(config.getoption("--visual"))
    setattr(config, "_cmm_visual", visual)
    if visual:
        setattr(config, "_cmm_visual_writer", None)
        setattr(config, "_cmm_visual_renderer", None)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Open one numbered trace and live renderer for the session."""
    config = session.config
    if not getattr(config, "_cmm_visual", False):
        return
    writer = TraceWriter(path=allocate_trace("pytest-visual")).open()
    renderer = LiveRenderer().open()
    setattr(config, "_cmm_visual_writer", writer)
    setattr(config, "_cmm_visual_renderer", renderer)
    _emit(config, "run_start", {"label": "pytest"})


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write the terminal event and close session resources."""
    config = session.config
    writer = getattr(config, "_cmm_visual_writer", None)
    if writer is None:
        return
    record = writer.finish({"status": str(exitstatus)})
    _render(config, record)
    renderer = getattr(config, "_cmm_visual_renderer", None)
    if renderer is not None:
        renderer.close()


@pytest.fixture(autouse=True)
def _visualize_decoder_calls(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe every Decoder.decode call while preserving user callbacks."""
    config = request.config
    if not getattr(config, "_cmm_visual", False):
        return
    original = Decoder.decode

    def observed(self: Decoder, *args: Any, **kwargs: Any) -> Any:
        trace_callback = kwargs.get("trace")
        callback = (
            trace_callback if trace_callback is not None
            else self.trace_observer
        )
        namespace = kwargs.get("namespace")
        is_name = (isinstance(namespace, tuple) and namespace
                   and namespace[0] == "names")
        _emit(config, "attempt_start", {
            "prompt_index": None,
            "phase": "selection" if is_name else "parameter", "path": [],
            "attempt": 1, "case_id": request.node.nodeid,
        })
        status = "success"

        def observer(event: TraceEvent) -> None:
            if callback is not None:
                callback(event)
            _emit(config, "token", {
                "step": event.step, "token_id": event.token_id,
                "raw_hex": event.raw.hex(),
                "allowed_count": event.allowed_count,
                "byte_count": event.byte_count, "complete": event.complete,
                "elapsed_seconds": event.elapsed_seconds,
            })

        kwargs["trace"] = observer
        try:
            return original(self, *args, **kwargs)
        except BaseException:
            status = "error"
            raise
        finally:
            _emit(config, "attempt_end", {"status": status})

    monkeypatch.setattr(Decoder, "decode", observed)


def _emit(
    config: pytest.Config, kind: str, data: dict[str, Any],
) -> TraceRecord | None:
    writer = getattr(config, "_cmm_visual_writer", None)
    if writer is None:
        return None
    record = cast(TraceWriter, writer).emit(kind, data)
    if record is not None:
        _render(config, record)
    return record


def _render(config: pytest.Config, record: TraceRecord) -> None:
    renderer = getattr(config, "_cmm_visual_renderer", None)
    if renderer is None:
        return
    # Display failures are intentionally best-effort; trace writes stay fatal.
    try:
        cast(LiveRenderer, renderer).observe(record)
    except Exception:
        pass
