"""Safe plain text rendering for diagnostic trace records."""

from typing import Any

from src.diagnostics.trace_schema import TraceRecord


def escape_terminal(value: object) -> str:
    """Render external text without allowing control sequences or new lines."""
    text = str(value)
    output: list[str] = []
    for char in text:
        code = ord(char)
        if code < 32 or 0x7F <= code <= 0x9F:
            output.append(f"\\x{code:02x}")
        else:
            output.append(char)
    return "".join(output)


def _field(data: dict[str, Any], key: str) -> str | None:
    if key not in data:
        return None
    return f"{key}={escape_terminal(data[key])}"


def render_record(record: TraceRecord) -> str:
    """Render one trace record as safe, compact plain text."""
    data = record.data
    parts = [f"seq={record.seq}", f"kind={record.kind}"]
    for key in (
        "phase", "path", "attempt", "model", "prompt_index", "step",
        "token_id", "raw_hex", "allowed_count", "byte_count", "complete",
        "elapsed_seconds", "status", "validation", "completion",
        "budget", "budgets", "typed_state_before", "typed_state_after",
    ):
        value = _field(data, key)
        if value is not None:
            parts.append(value)
    return _bound_line(" | ".join(parts))


def render_view(record: TraceRecord, state: Any) -> str:
    """Render a record with its bounded reconstructed attempt context."""
    parts = [render_record(record)]
    if state.model is not None:
        parts.append(f"current_model={escape_terminal(state.model)}")
    if state.attempt:
        for key in ("phase", "path", "attempt", "prompt_index"):
            if key in state.attempt:
                parts.append(
                    f"current_{key}={escape_terminal(state.attempt[key])}"
                )
    if state.text:
        parts.append(f"decoded={escape_terminal(state.text)}")
    if state.pending_hex:
        parts.append(f"pending_utf8_hex={escape_terminal(state.pending_hex)}")
    if state.decode_error_hex:
        parts.append(
            "utf8_decode_error_hex="
            f"{escape_terminal(state.decode_error_hex)}"
        )
    if state.frames:
        parts.append(f"frames={escape_terminal(state.frames)}")
    return _bound_line(" | ".join(parts))


def _bound_line(line: str, limit: int = 1200) -> str:
    """Keep terminal output to a manageable width-independent size."""
    if len(line) <= limit:
        return line
    return f"{line[:limit - 3]}..."
