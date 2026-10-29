"""Command-line function calling with complete-batch atomic output."""

import argparse
import sys

from src.config import RunSettings
from src.decoding.failures import FailureRecord
from src.llm.profiles import PROFILES, get_profile
from src.io.reader import load_function_definitions, load_prompt_items
from src.io.writer import write_json_file
from src.utils.logging import log_error
from src.diagnostics.performance import measure
from src.diagnostics.trace_io import TraceWriter
from src.diagnostics.capture import attach_capture, start_capture
from src.diagnostics.trace_paths import allocate_trace


def _run(settings: RunSettings, failures: list[FailureRecord]) -> None:
    """Validate and publish a complete batch, retaining recovery failures."""
    label = (
        "baseline_11"
        if settings.input_path == RunSettings().input_path
        else settings.input_path.stem
    )
    trace_path = settings.trace_path or allocate_trace(label)
    protected = {
        settings.input_path.resolve(),
        settings.functions_path.resolve(),
        settings.output_path.resolve(),
    }
    protected.update(
        path.resolve()
        for path in (
            settings.diagnostics_path,
            settings.performance_path,
        )
        if path is not None
    )
    if trace_path.resolve() in protected:
        if settings.trace_path is None:
            trace_path.unlink(missing_ok=True)
        raise ValueError("Trace must use a separate output file.")
    renderer = None
    if settings.visual:
        from src.visualization.live import LiveRenderer

        renderer = LiveRenderer().open()
    writer: TraceWriter | None = None
    try:
        writer = TraceWriter(
            path=trace_path,
            observer=renderer.observe if renderer else None,
        ).open()
    except BaseException:
        if renderer is not None:
            renderer.close()
        raise
    assert writer is not None
    print(f"Trace: {trace_path}", file=sys.stderr)
    try:
        start_capture(writer, settings)
        prompts = load_prompt_items(settings.input_path)
        functions = load_function_definitions(
            settings.functions_path,
            allow_nested=settings.profile == "bonus",
            schema_limits=settings.schema_limits,
        )
        if prompts and not functions:
            raise ValueError("No available functions for the input prompts.")
        if not prompts:
            writer.finish(
                {
                    "status": "completed",
                    "validation": "passed",
                    "publication": "pending",
                }
            )
            write_json_file(settings.output_path, [])
            return
        from src.pipeline.function_caller import create_caller

        caller = create_caller(functions, settings)
        attach_capture(caller, writer)
        if settings.diagnostics_path is not None:
            caller.failure_observer = failures.append
        calls = caller.run(prompts)
        # Flush/close diagnostic IO before publishing the atomic batch.
        if writer is not None:
            writer.finish(
                {
                    "status": "completed",
                    "validation": "passed",
                    "publication": "pending",
                }
            )
            writer = None
        with measure(getattr(caller, "performance", None), "output_io"):
            write_json_file(
                settings.output_path,
                [call.model_dump() for call in calls],
            )
        if settings.performance_path is not None:
            cache = caller.decoder.cache
            write_json_file(
                settings.performance_path,
                {
                    "timings": (
                        caller.performance.report()
                        if caller.performance is not None
                        else {}
                    ),
                    "cache": {
                        "entries": cache.entry_count,
                        "resident_bytes": cache.resident_bytes,
                        "peak_bytes": cache.peak_bytes,
                        "hits": cache.hits,
                        "misses": cache.misses,
                        "evictions": cache.evictions,
                        "oversize_bypasses": cache.oversize_bypasses,
                    },
                    "publication": "completed",
                },
            )
    except BaseException:
        if writer is not None and writer.is_open:
            try:
                writer.finish({"status": "failed", "publication": "aborted"})
            except OSError:
                # Keep the original failure visible if a failed destination
                # cannot record its own failure event either.
                pass
            finally:
                writer.close()
        raise
    finally:
        if renderer is not None:
            renderer.close()


def main() -> int:
    """Validate inputs, decode a complete batch, and report failures.

    Returns:
        Zero on success, one for a handled error, or 130 on interruption.
    """
    parser = argparse.ArgumentParser(description="Prompts to function calls.")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Functions definition file path.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Input file path.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="Output file path.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=1536)
    parser.add_argument(
        "--profile", choices=("mandatory", "bonus"), default="mandatory"
    )
    parser.add_argument("--tokenizer", choices=("sdk", "local"), default="sdk")
    parser.add_argument(
        "--recovery", choices=("none", "retry"), default="none"
    )
    parser.add_argument(
        "--model", choices=tuple(PROFILES), default="Qwen/Qwen3-0.6B"
    )
    parser.add_argument("--model-snapshot")
    parser.add_argument("--diagnostics-output")
    parser.add_argument("--performance-output")
    parser.add_argument("--trace-output")
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Show live constrained generation in the terminal.",
    )
    args = parser.parse_args()

    settings: RunSettings | None = None
    failures: list[FailureRecord] = []
    status = 0
    message: str | None = None
    try:
        settings = RunSettings(
            functions_path=args.functions_definition,
            input_path=args.input,
            output_path=args.output,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            max_context_tokens=args.max_context_tokens,
            profile=args.profile,
            tokenizer=args.tokenizer,
            recovery=args.recovery,
            model_name=args.model,
            model_snapshot=args.model_snapshot,
            diagnostics_path=args.diagnostics_output,
            performance_path=args.performance_output,
            trace_path=args.trace_output,
            visual=args.visual,
        )
        _run(settings, failures)
    except KeyboardInterrupt:
        status, message = 130, "Interrupted."
    except Exception as error:
        status, message = 1, str(error)
    if message is not None:
        log_error(message)
    if settings is not None and settings.diagnostics_path is not None:
        try:
            model = get_profile(settings.model_name)
            write_json_file(
                settings.diagnostics_path,
                {
                    "exit_status": status,
                    "error": message,
                    "model_id": model.model_id,
                    "revision": model.revision,
                    "snapshot_pinned": settings.profile == "bonus",
                    "recovery": settings.recovery,
                    "failures": [
                        item.model_dump(mode="json") for item in failures
                    ],
                },
            )
        except Exception as error:
            log_error(f"Cannot write diagnostics: {error}")
            if status == 0:
                status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
