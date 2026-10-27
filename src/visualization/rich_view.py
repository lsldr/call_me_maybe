"""Optional, literal-text Rich tables for saved generation traces."""

from src.diagnostics.trace_schema import TraceRecord
from src.visualization.plain import escape_terminal
from src.visualization.replay import ReplayState


def render_rich(
    record: TraceRecord,
    state: ReplayState | None = None,
    *,
    color: bool = True,
) -> str:
    """Render known trace fields as safe table rows when Rich is installed."""
    from src.visualization.plain import render_view

    view = state or ReplayState()
    plain = render_view(record, view)
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        return plain
    if not color:
        return plain

    data = record.data
    attempt = view.attempt
    fields: list[tuple[str, object]] = [
        ("run", view.model),
        ("model", data.get("model", view.model)),
        ("phase", data.get("phase", attempt.get("phase"))),
        ("path", data.get("path", attempt.get("path"))),
        ("attempt", data.get("attempt", attempt.get("attempt"))),
        ("step", data.get("step")),
        ("token ID", data.get("token_id")),
        ("raw hex", data.get("raw_hex")),
        ("decoded text", view.text or None),
        ("pending UTF-8", view.pending_hex or None),
        ("allowed count", data.get("allowed_count")),
        ("typed state before", data.get("typed_state_before")),
        ("typed state after", data.get("typed_state_after")),
        ("frames", view.frames or None),
        ("token/byte budget", data.get("budgets", data.get("budget"))),
        ("token limit", data.get("max_steps")),
        ("byte limit", data.get("max_value_bytes")),
        ("grammar complete", data.get("complete")),
        ("step seconds", data.get("elapsed_seconds")),
        ("validated value", data.get("value")),
        ("token bytes", data.get("byte_count")),
        ("final validation", data.get("validation")),
        ("status", data.get("status")),
    ]
    table = Table(show_header=False, expand=True)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for label, value in fields:
        table.add_row(
            Text(label),
            Text("unavailable" if value is None else escape_terminal(value)),
        )
    console = Console(
        record=True, force_terminal=True, color_system="standard",
    )
    with console.capture() as capture:
        console.print(table, markup=False, highlight=False, soft_wrap=True)
    return capture.get().rstrip("\n")
