"""Incremental terminal rendering for live generation traces."""

import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr

from src.diagnostics.trace_schema import TraceRecord
from src.visualization.plain import render_record
from src.visualization.replay import ReplayState


class LiveRenderer(BaseModel):
    """Render trace events as they arrive, with a safe plain fallback."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stream: Any = sys.stderr
    terminal: bool | None = None
    _state: ReplayState = PrivateAttr(default_factory=ReplayState)
    _live: Any = PrivateAttr(default=None)
    _rich_text: Any = PrivateAttr(default=None)
    _opened: bool = PrivateAttr(default=False)
    _index: int = PrivateAttr(default=-1)

    def open(self) -> "LiveRenderer":
        """Start terminal updating when Rich and a TTY are available."""
        if self._opened:
            return self
        self._opened = True
        if not self._is_terminal():
            return self
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.text import Text

            console = Console(
                file=self.stream, force_terminal=True,
                color_system="standard",
            )
            self._live = Live(
                Text("Waiting for trace events"), console=console,
                auto_refresh=False, transient=False,
            )
            self._rich_text = Text
            self._live.start()
        except ImportError:
            self._live = None
            self._rich_text = None
        except Exception:
            # Rendering must not interrupt model inference.
            self._live = None
            self._rich_text = None
        return self

    def observe(self, record: TraceRecord) -> None:
        """Apply and immediately render one event without retaining history."""
        if not self._opened:
            self.open()
        if record.kind == "run_start":
            self._state = ReplayState()
        self._index = max(self._index + 1, record.seq)
        try:
            self._state.apply(record, self._index)
            if self._live is not None:
                from src.visualization.rich_view import render_rich

                text_type = self._rich_text
                rendered = render_rich(record, self._state)
                self._live.update(
                    text_type.from_ansi(rendered), refresh=True
                )
            else:
                self.stream.write(self._plain_event(record) + "\n")
                self.stream.flush()
        except Exception:
            # A broken/closed terminal is observational only.
            self._stop_live()

    def close(self) -> None:
        """Stop the Rich display, suppressing terminal cleanup failures."""
        self._stop_live()
        self._opened = False

    def __enter__(self) -> "LiveRenderer":
        return self.open()

    def __exit__(self, exc_type: object, exc: object,
                 traceback: object) -> Literal[False]:
        self.close()
        return False

    def _is_terminal(self) -> bool:
        if self.terminal is not None:
            return self.terminal
        try:
            return bool(self.stream.isatty())
        except (AttributeError, OSError):
            return False

    def _plain_event(self, record: TraceRecord) -> str:
        from src.visualization.plain import render_view

        # The view includes decoded bytes, attempt context, and grammar frames.
        return render_view(record, self._state) or render_record(record)

    def _stop_live(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
        self._live = None
        self._rich_text = None
