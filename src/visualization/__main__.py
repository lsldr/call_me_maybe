"""Command line entry point for the offline trace viewer."""

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

from src.diagnostics.trace_io import TraceReader
from src.diagnostics.trace_schema import TraceRecord
from src.visualization.plain import escape_terminal, render_view
from src.visualization.replay import ReplayState


def select_trace(
    trace_dir: Path,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> Path | None:
    """Prompt for one of the newest numbered traces in *trace_dir*.

    Returns ``None`` when the user quits or reaches EOF. Raises
    ``FileNotFoundError`` when the directory contains no trace files.
    """
    traces = list(trace_dir.glob("*.jsonl"))
    traces.sort(key=_trace_sort_key)
    if not traces:
        raise FileNotFoundError(f"No trace files found in {trace_dir}")
    output_fn("Available traces:")
    for index, trace in enumerate(traces, start=1):
        output_fn(f"{index}. {escape_terminal(trace.name)}")
    while True:
        try:
            selection = input_fn("Select a trace number (or q to quit): ")
        except EOFError:
            return None
        selection = selection.strip()
        if selection.lower() == "q":
            return None
        if not selection.isdecimal():
            output_fn("Enter a listed number or q to quit.")
            continue
        index = int(selection)
        if index < 1 or index > len(traces):
            output_fn("That trace number is out of range.")
            continue
        return traces[index - 1]


def _trace_sort_key(path: Path) -> tuple[int, str]:
    """Sort numbered traces newest first, then by filename.

    Args:
        path: Trace path whose leading numeric prefix determines order.

    Returns:
        Sort key with descending numeric prefix and filename tie-breaker.
    """
    match = re.match(r"^(\d+)_", path.name)
    number = int(match.group(1)) if match is not None else 0
    return -number, path.name


def main() -> int:
    """Run the offline trace viewer command-line interface.

    Returns:
        Process status code: zero on success, nonzero on read/render errors.
    """
    parser = argparse.ArgumentParser(
        description="View a diagnostic trace offline"
    )
    parser.add_argument("trace", nargs="?")
    parser.add_argument(
        "--trace-dir", type=Path, default=Path("data/traces"),
        help="Directory used to select a trace when no path is supplied.",
    )
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Read complete records before an incomplete final trace line.",
    )
    args = parser.parse_args()
    trace_path = Path(args.trace) if args.trace is not None else None
    if trace_path is None:
        try:
            trace_path = select_trace(
                args.trace_dir, input_fn=input, output_fn=print,
            )
        except FileNotFoundError as error:
            print(f"Cannot select trace: {error}", file=sys.stderr)
            return 1
        if trace_path is None:
            return 0
    try:
        reader = TraceReader(
            path=trace_path, allow_partial=args.allow_partial,
        ).open()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (OSError, ValueError) as error:
        print(f"Cannot read trace: {error}", file=sys.stderr)
        return 1
    if reader.partial:
        print(
            "Warning: trace is partial or truncated; only complete records "
            "are available.",
            file=sys.stderr,
        )
    if args.interactive:
        from src.visualization.replay import interactive

        try:
            interactive(reader)
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr)
            return 130
        except (OSError, ValueError) as error:
            print(f"Cannot render trace: {error}", file=sys.stderr)
            return 1
        return 0
    use_rich = not args.plain and not args.no_color and sys.stdout.isatty()
    renderer: Callable[[TraceRecord, ReplayState], str]
    if use_rich:
        from src.visualization.rich_view import render_rich

        renderer = render_rich
    else:
        renderer = render_view
    state = ReplayState()
    try:
        for index, record in enumerate(reader):
            state.apply(record, index)
            print(renderer(record, state))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (OSError, ValueError) as error:
        print(f"Cannot render trace: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
