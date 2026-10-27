"""Capture one production-path constrained function call."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import Any, Literal
from unittest.mock import patch

from src.config import RunSettings
from src.diagnostics.capture import attach_capture, start_capture
from src.diagnostics.trace_io import TraceWriter
from src.diagnostics.trace_paths import allocate_trace
from src.evaluation.benchmark import (
    _execute_case,
    _finish_trace_on_error,
    _offline_environment,
    _write_json,
)
from src.evaluation.cases import load_cases
from src.io.reader import load_function_definitions
from src.pipeline.function_caller import create_caller

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")


def _guard_sdk_tokenizer(stack: ExitStack, tokenizer: str) -> None:
    """Ensure local demo mode never calls SDK encode or decode."""
    if tokenizer != "local":
        return
    from src.llm.model_wrapper import ModelWrapper

    stack.enter_context(patch.object(
        ModelWrapper, "encode",
        side_effect=AssertionError("SDK encode called in local demo mode"),
    ))
    stack.enter_context(patch.object(
        ModelWrapper, "decode",
        side_effect=AssertionError("SDK decode called in local demo mode"),
    ))


def run_demo(
    output: Path,
    *,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    tokenizer: Literal["sdk", "local"] = "local",
) -> Path:
    """Generate one screened case and save context plus token evidence."""
    _offline_environment()
    output.mkdir(parents=True, exist_ok=True)
    case = load_cases("benchmark")[0]
    functions = load_function_definitions(DEFAULT_FUNCTIONS)
    settings = RunSettings(
        device=device,
        profile="bonus" if tokenizer == "local" else "mandatory",
        tokenizer=tokenizer,
        optimize=True,
        functions_path=DEFAULT_FUNCTIONS,
    )
    stack = ExitStack()
    writer: TraceWriter | None = None
    try:
        _guard_sdk_tokenizer(stack, tokenizer)
        trace_path = allocate_trace("demo")
        settings = settings.model_copy(update={"trace_path": trace_path})
        writer = TraceWriter(path=trace_path)
        writer.open()
        start_capture(writer, settings)
        caller = create_caller(functions, settings)
        attach_capture(caller, writer)
        result = _execute_case(caller, case, functions, 0)
        decode_evidence: list[dict[str, Any]] = []
        if tokenizer == "local":
            local_tokenizer = getattr(caller.encode, "__self__", None)
            decode = getattr(local_tokenizer, "decode", None)
            if not callable(decode):
                decode_evidence.append({
                    "error": "local tokenizer public decode is unavailable",
                    "matches_trace_bytes": False,
                })
            else:
                for attempt in result["attempts"]:
                    token_ids = attempt["token_ids"]
                    raw_hex = "".join(attempt["raw_hex"])
                    evidence: dict[str, Any] = {
                        "phase": attempt["phase"],
                        "path": attempt["path"],
                        "attempt": attempt["attempt"],
                        "token_ids": token_ids,
                        "raw_hex": raw_hex,
                    }
                    try:
                        decoded = decode(token_ids)
                        decoded_hex = decoded.encode("utf-8").hex()
                        evidence["decoded_text"] = decoded
                        evidence["decoded_hex"] = decoded_hex
                        evidence["matches_trace_bytes"] = (
                            decoded_hex == raw_hex
                        )
                    except Exception as error:
                        evidence["decode_error"] = (
                            f"{type(error).__name__}: {error}"
                        )
                        evidence["matches_trace_bytes"] = False
                    decode_evidence.append(evidence)
        passed = bool(result["valid"] and result["accepted_match"])
        if tokenizer == "local":
            passed = passed and bool(decode_evidence) and all(
                evidence.get("matches_trace_bytes") is True
                for evidence in decode_evidence
            )
        writer.finish({
            "status": "completed" if passed else "failed",
            "validation": "passed" if passed else "failed",
            "message": result.get("error"),
        })
        artifact = {
            "case_id": case.id,
            "prompt": case.prompt,
            "function_definitions": [
                function.model_dump(mode="json")
                for function in functions.values()
            ],
            "expected": case.expected.model_dump(mode="json"),
            "result": result,
            "local_decode_evidence": decode_evidence,
            "local_decode_matches_trace": (
                bool(decode_evidence) and all(
                    item.get("matches_trace_bytes") is True
                    for item in decode_evidence
                )
                if tokenizer == "local" else None
            ),
            "trace_path": str(trace_path),
            "tokenizer": tokenizer,
            "device": device,
            "passed": passed,
        }
        path = output / "demo.json"
        _write_json(path, artifact)
        return path
    except BaseException as error:
        if writer is not None:
            _finish_trace_on_error(writer, error)
        raise
    finally:
        stack.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "mps"), default="cpu",
    )
    parser.add_argument(
        "--tokenizer", choices=("sdk", "local"), default="local",
    )
    args = parser.parse_args()
    try:
        path = run_demo(
            args.output, device=args.device, tokenizer=args.tokenizer
        )
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"demo failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(path)
    return 0 if artifact.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
