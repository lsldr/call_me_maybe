"""Command-line function calling with complete-batch atomic output."""

import argparse

from src.config import RunSettings
from src.decoding.failures import FailureRecord
from src.io.reader import load_function_definitions, load_prompt_items
from src.io.writer import write_json_file
from src.utils.logging import log_error

PROFILES = {
    "Qwen/Qwen3-0.6B",
    "HF/SmolLM2-360M",
    "HF/SmalLM2-135M",
    "HF/SmalLM1-135M",
}


def _run(settings: RunSettings, failures: list[FailureRecord]) -> None:
    """Validate and publish a complete batch, retaining recovery failures."""
    try:
        prompts = load_prompt_items(settings.input_path)
        functions = load_function_definitions(
            settings.functions_path,
            allow_nested=settings.profile == "bonus",
            schema_limits=settings.schema_limits,
        )
        if prompts and not functions:
            raise ValueError("No available functions for the input prompts.")
        if not prompts:
            write_json_file(settings.output_path, [])
            return

        from src.pipeline.function_caller import create_caller

        caller = create_caller(functions, settings)
        calls = caller.run(prompts)
        write_json_file(
            settings.output_path,
            [call.model_dump() for call in calls],
        )
    except BaseException:
        raise


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
    parser.add_argument("--tokenizer", choices=("sdk", "local"), default="sdk")
    parser.add_argument("--model", choices=PROFILES, default="Qwen/Qwen3-0.6B")
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
            tokenizer=args.tokenizer,
            model_name=args.model,
        )
        _run(settings, failures)
    except KeyboardInterrupt:
        status, message = 130, "Interrupted."
    except Exception as error:
        status, message = 1, str(error)
    if message is not None:
        log_error(message)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
