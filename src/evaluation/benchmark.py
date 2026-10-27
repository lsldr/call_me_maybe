"""Run one constrained generation pass over ten demonstration cases."""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Literal

from src.config import RunSettings
from src.diagnostics.capture import attach_capture
from src.diagnostics.trace_io import TraceWriter
from src.evaluation.cases import EvalCase, load_cases
from src.io.reader import load_function_definitions, validate_function_calls
from src.models.function_def import FunctionDef
from src.models.prompt import PromptItem
from src.pipeline.function_caller import FunctionCaller, create_caller

BENCHMARK_CASE_COUNT = 10
DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
ORIGINAL_PROMPTS = Path("data/evaluation/original_prompts.json")


def _offline_environment() -> None:
    """Prevent Hugging Face components from downloading model assets."""
    for name in (
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
    ):
        os.environ[name] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finish_trace_on_error(writer: TraceWriter, error: BaseException) -> None:
    """Try to close a failed trace without replacing the original error."""
    if not writer.is_open:
        return
    try:
        writer.finish({
            "status": "failed",
            "message": f"{type(error).__name__}: {error}",
        })
    except Exception as finish_error:
        error.args = (*error.args, (
            "Could not write trace failure footer: "
            f"{type(finish_error).__name__}: {finish_error}"
        ))


def _original_prompt_texts() -> set[str]:
    """Read copied prompts without depending on expected-call records."""
    rows = json.loads(ORIGINAL_PROMPTS.read_text(encoding="utf-8"))
    return {row["prompt"] for row in rows}


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize accuracy with every failed row kept in the denominator."""
    total = len(results)
    valid = sum(result.get("valid") is True for result in results)
    exact = sum(result.get("exact_match") is True for result in results)
    accepted = sum(result.get("accepted_match") is True for result in results)
    function_correct = sum(
        result.get("function_correct") is True for result in results
    )
    return {
        "total": total,
        "valid": valid,
        "exact": exact,
        "accepted": accepted,
        "function_correct": function_correct,
        "failed": total - valid,
        "mismatched": sum(
            result.get("valid") is True
            and result.get("accepted_match") is not True
            for result in results
        ),
        "valid_percent": (100.0 * valid / total if total else 0.0),
        "exact_accuracy_percent": (
            100.0 * exact / total if total else 0.0
        ),
        "accepted_accuracy_percent": (
            100.0 * accepted / total if total else 0.0
        ),
        "function_accuracy_percent": (
            100.0 * function_correct / total if total else 0.0
        ),
    }


def regex_behavior_match(
    expected: dict[str, Any], actual: dict[str, Any], probes: list[str],
) -> bool:
    """Compare regex spans and substitutions on source plus finite probes."""
    if (
        expected.get("source_string") != actual.get("source_string")
        or expected.get("replacement") != actual.get("replacement")
    ):
        return False
    expected_pattern = expected.get("regex")
    actual_pattern = actual.get("regex")
    source = expected.get("source_string")
    if (
        not isinstance(expected_pattern, str)
        or not isinstance(actual_pattern, str)
        or not isinstance(source, str)
    ):
        return False
    try:
        expected_re = re.compile(expected_pattern)
        actual_re = re.compile(actual_pattern)
        for text in [source, *probes]:
            expected_spans = [
                match.span() for match in expected_re.finditer(text)
            ]
            actual_spans = [match.span() for match in actual_re.finditer(text)]
            if expected_spans != actual_spans:
                return False
            replacement = str(expected["replacement"])
            if expected_re.sub(replacement, text) != actual_re.sub(
                replacement, text,
            ):
                return False
    except (re.error, KeyError, TypeError, ValueError):
        return False
    return True


def _execute_case(
    caller: FunctionCaller,
    case: EvalCase,
    functions: dict[str, FunctionDef],
    index: int,
) -> dict[str, Any]:
    """Run one case, retaining selected name even if arguments fail."""
    contexts: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    selected: str | None = None
    current_attempt: dict[str, Any] | None = None
    original_encode = caller.encode
    original_context = caller.context_observer
    original_selection = caller.selection_observer
    original_attempt = caller.attempt_observer
    original_trace = caller.decoder.trace_observer
    pending_context: dict[str, Any] | None = None

    def context_observer(phase: str, text: str) -> None:
        nonlocal pending_context
        pending_context = {"phase": phase, "text": text}
        contexts.append(pending_context)
        if original_context is not None:
            original_context(phase, text)

    def encode(text: str) -> list[int]:
        nonlocal pending_context
        ids = original_encode(text)
        if pending_context is not None:
            pending_context["input_ids"] = ids
            pending_context = None
        return ids

    def selection_observer(name: str) -> None:
        nonlocal selected
        selected = name
        if original_selection is not None:
            original_selection(name)

    def attempt_observer(kind: str, data: dict[str, Any]) -> None:
        nonlocal current_attempt
        if kind == "attempt_start":
            current_attempt = {
                "phase": data.get("phase"), "path": data.get("path"),
                "attempt": data.get("attempt"), "token_ids": [],
                "raw_hex": [],
            }
            attempts.append(current_attempt)
        if original_attempt is not None:
            original_attempt(kind, data)

    def token_observer(event: Any) -> None:
        if current_attempt is not None:
            current_attempt["token_ids"].append(event.token_id)
            current_attempt["raw_hex"].append(event.raw.hex())
        if original_trace is not None:
            original_trace(event)

    caller.encode = encode
    caller.context_observer = context_observer
    caller.selection_observer = selection_observer
    caller.attempt_observer = attempt_observer
    caller.decoder.trace_observer = token_observer
    started = time.perf_counter()
    row: dict[str, Any] = {
        "id": case.id,
        "category": case.category,
        "prompt": case.prompt,
        "expected": case.expected.model_dump(mode="json"),
        "selected_name": None,
        "function_correct": False,
        "actual": None,
        "valid": False,
        "exact_match": False,
        "behavior_match": None,
        "accepted_match": False,
        "elapsed_seconds": 0.0,
        "contexts": contexts,
        "attempts": attempts,
    }
    try:
        call = caller.call(PromptItem(prompt=case.prompt), prompt_index=index)
        actual = call.model_dump(mode="json")
        row["actual"] = actual
        row["selected_name"] = selected
        validated = validate_function_calls(
            [actual], functions, [PromptItem(prompt=case.prompt)],
        )[0]
        row["valid"] = True
        row["function_correct"] = selected == case.expected.name
        row["exact_match"] = (
            selected == case.expected.name
            and validated.name == case.expected.name
            and validated.parameters == case.expected.parameters
        )
        is_regex = (
            case.expected.name == "fn_substitute_string_with_regex"
            and "regex" in case.expected.parameters
        )
        behavior_match = (
            regex_behavior_match(
                case.expected.parameters, validated.parameters,
                list(getattr(case, "probes", [])),
            ) if is_regex else None
        )
        row["behavior_match"] = behavior_match
        row["behavior_probes"] = list(getattr(case, "probes", []))
        row["accepted_match"] = (
            row["exact_match"] is True
            or (
                is_regex and behavior_match is True
                and selected == case.expected.name
                and validated.name == case.expected.name
            )
        )
        if not row["accepted_match"]:
            row["error"] = "Validated call does not match expected output."
    except OSError:
        row["selected_name"] = selected
        row["function_correct"] = selected == case.expected.name
        raise
    except Exception as error:
        row["selected_name"] = selected
        row["function_correct"] = selected == case.expected.name
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        row["elapsed_seconds"] = time.perf_counter() - started
        caller.encode = original_encode
        caller.context_observer = original_context
        caller.selection_observer = original_selection
        caller.attempt_observer = original_attempt
        caller.decoder.trace_observer = original_trace
    return row


def _benchmark_settings(
    *,
    device: Literal["cpu", "cuda", "mps"],
    tokenizer: Literal["sdk", "local"],
    visual: bool,
) -> RunSettings:
    return RunSettings(
        device=device,
        profile="bonus" if tokenizer == "local" else "mandatory",
        tokenizer=tokenizer,
        optimize=True,
        visual=visual,
        functions_path=DEFAULT_FUNCTIONS,
    )


def run_benchmark(
    output: Path,
    *,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    tokenizer: Literal["sdk", "local"] = "sdk",
    visual: bool = False,
) -> dict[str, Any]:
    """Run the ten benchmark prompts once with one loaded model."""
    _offline_environment()
    output.mkdir(parents=True, exist_ok=True)
    from src.diagnostics.capture import start_capture
    from src.diagnostics.trace_paths import allocate_trace

    renderer: Any = None
    observer: Any = None
    if visual:
        from src.visualization.live import LiveRenderer

        renderer = LiveRenderer()
        renderer.open()
        observer = renderer.observe
    rows: list[dict[str, Any]] = []
    writer: TraceWriter | None = None
    try:
        trace_path = allocate_trace("benchmark_10")
        writer = TraceWriter(path=trace_path, observer=observer)
        writer.open()
        settings = _benchmark_settings(
            device=device, tokenizer=tokenizer, visual=visual,
        ).model_copy(update={
            "trace_path": trace_path,
        })
        start_capture(writer, settings)

        cases = load_cases("benchmark")
        if len(cases) != BENCHMARK_CASE_COUNT:
            raise ValueError(
                f"benchmark suite must contain exactly "
                f"{BENCHMARK_CASE_COUNT} cases, got {len(cases)}"
            )
        original_prompts = _original_prompt_texts()
        if any(case.prompt in original_prompts for case in cases):
            raise ValueError(
                "benchmark prompts must not duplicate original cases"
            )
        functions = load_function_definitions(DEFAULT_FUNCTIONS)
        caller = create_caller(functions, settings)
        attach_capture(caller, writer)
        for index, case in enumerate(cases):
            row = _execute_case(
                caller, case, functions, index,
            )
            rows.append(row)
            _write_json(output / "benchmark.json", {
                "suite": "benchmark",
                "case_count": BENCHMARK_CASE_COUNT,
                "completed": len(rows),
                "function_definitions": [
                    item.model_dump(mode="json")
                    for item in functions.values()
                ],
                "prompts": [item.prompt for item in cases],
                "expected": [
                    item.expected.model_dump(mode="json") for item in cases
                ],
                "results": rows,
                "summary": summarize_results(rows),
                "trace_path": str(trace_path),
                "tokenizer": tokenizer,
                "device": device,
                "visual": visual,
            })
            print(
                f"benchmark {len(rows)}/{len(cases)} {case.id} "
                f"valid={row['valid']} exact={row['exact_match']}",
                flush=True,
            )
        summary = summarize_results(rows)
        passed = (
            summary["valid"] == len(cases)
            and summary["accepted"] == len(cases)
        )
        writer.finish({
            "status": "completed" if passed else "failed",
            "validation": "passed" if passed else "failed",
            "summary": summary,
        })
        report = {
            "suite": "benchmark",
            "case_count": len(cases),
            "function_definitions": [
                item.model_dump(mode="json") for item in functions.values()
            ],
            "prompts": [case.prompt for case in cases],
            "expected": [
                case.expected.model_dump(mode="json") for case in cases
            ],
            "results": rows,
            "summary": summary,
            "trace_path": str(trace_path),
            "tokenizer": tokenizer,
            "device": device,
            "visual": visual,
            "passed": passed,
        }
        _write_json(output / "benchmark.json", report)
        return report
    except BaseException as error:
        if writer is not None:
            _finish_trace_on_error(writer, error)
        raise
    finally:
        if renderer is not None:
            renderer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "mps"), default="cpu",
    )
    parser.add_argument(
        "--tokenizer", choices=("sdk", "local"), default="sdk",
    )
    parser.add_argument("--visual", action="store_true")
    args = parser.parse_args()
    try:
        report = run_benchmark(
            args.output, device=args.device, tokenizer=args.tokenizer,
            visual=args.visual,
        )
    except Exception as error:
        print(
            f"benchmark failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
