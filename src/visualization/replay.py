"""Bounded state reconstruction and navigation for offline traces."""

import codecs
from typing import Any
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr

from src.diagnostics.trace_io import TraceReader
from src.diagnostics.trace_schema import TraceRecord


class ReplayState(BaseModel):
    """Small current-attempt view reconstructed from indexed trace records."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    index: int = -1
    current_attempt_bytes: bytes = b""
    pending_hex: str = ""
    text: str = ""
    decode_error_hex: str = ""
    model: str | None = None
    attempt: dict[str, JsonValue] = Field(default_factory=dict)
    frames: list[dict[str, JsonValue]] = Field(default_factory=list)
    max_attempt_bytes: int = Field(default=65_536, ge=0)
    _decoder: Any = PrivateAttr(
        default_factory=lambda: codecs.getincrementaldecoder("utf-8")("strict")
    )

    def apply(self, record: TraceRecord, index: int) -> None:
        """Apply one record, retaining only bounded current attempt bytes."""
        self.index = index
        if record.kind == "run_start":
            model = record.data.get("model")
            self.model = model if isinstance(model, str) else None
            self.frames = [{"kind": "run", "model": self.model}]
        elif record.kind == "attempt_start":
            self.current_attempt_bytes = b""
            self.pending_hex = ""
            self.text = ""
            self.decode_error_hex = ""
            self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
            self.attempt = dict(record.data)
            self.frames = [{"kind": "run", "model": self.model},
                           {"kind": "attempt", **self.attempt}]
        elif record.kind == "token":
            chunk = bytes.fromhex(str(record.data["raw_hex"]))
            remaining = self.max_attempt_bytes - len(
                self.current_attempt_bytes
            )
            if remaining > 0:
                kept = chunk[:remaining]
                self.current_attempt_bytes += kept
                buffered = bytes(self._decoder.getstate()[0])
                try:
                    self.text += self._decoder.decode(kept, final=False)
                except UnicodeDecodeError:
                    self.decode_error_hex = (buffered + kept).hex()
                    self._decoder = codecs.getincrementaldecoder("utf-8")(
                        "strict"
                    )
            self.pending_hex = bytes(self._decoder.getstate()[0]).hex()
            self.frames = self.frames[:2]
            for name in ("typed_state_before", "typed_state_after"):
                value = record.data.get(name)
                if isinstance(value, dict):
                    self.frames.append({"kind": name, **value})


def state_at(reader: TraceReader, index: int) -> ReplayState:
    """Reconstruct current attempt incrementally up to an indexed record."""
    if index < 0 or index >= len(reader):
        raise IndexError(index)
    state = ReplayState()
    attempt_start = 0
    for position in range(index, -1, -1):
        record = reader.read(position)
        if record.kind == "attempt_start":
            attempt_start = position
            break
        if record.kind == "run_start":
            attempt_start = position
            break
    run_start = 0
    for position in range(attempt_start, -1, -1):
        if reader.read(position).kind == "run_start":
            run_start = position
            break
    state.apply(reader.read(run_start), run_start)
    for position in range(max(attempt_start, run_start + 1), index + 1):
        state.apply(reader.read(position), position)
    return state


def interactive(
    reader: TraceReader,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    sleep_fn: Callable[[float], None] | None = None,
    speed: float = 1.0,
) -> None:
    """Run line-oriented navigation commands without raw terminal handling."""
    import math
    import time

    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be positive and finite")
    pause = sleep_fn or time.sleep
    index = 0 if len(reader) else -1
    while index >= 0:
        output_fn(__render(reader, index))
        try:
            command = input_fn(
                "[next/prev/goto N/run/speed N/quit] "
            ).strip().split()
        except EOFError:
            return
        if not command or command[0] in ("next", "n"):
            index = min(index + 1, len(reader) - 1)
        elif command[0] in ("prev", "p"):
            index = max(0, index - 1)
        elif command[0] == "goto" and len(command) == 2:
            try:
                index = max(0, min(int(command[1]), len(reader) - 1))
            except ValueError:
                output_fn("invalid index")
        elif command[0] == "speed" and len(command) == 2:
            try:
                candidate = float(command[1])
                if not math.isfinite(candidate) or candidate <= 0:
                    raise ValueError
                speed = candidate
            except ValueError:
                output_fn("speed must be positive and finite")
        elif command[0] == "run":
            for next_index in range(index + 1, len(reader)):
                pause(1.0 / speed)
                index = next_index
                output_fn(__render(reader, index))
        elif command[0] == "quit":
            return


def __render(reader: TraceReader, index: int) -> str:
    """Render one indexed trace record with its reconstructed state.

    Args:
        reader: Indexed source trace.
        index: Record index to render.

    Returns:
        Plain-text view of the record and state at that position.
    """
    from src.visualization.plain import render_view

    return render_view(reader.read(index), state_at(reader, index))
