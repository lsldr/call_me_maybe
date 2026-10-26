"""Command-line function calling with complete-batch atomic output."""

import argparse

from src.config import RunSettings
from src.io.reader import load_function_definitions, load_prompt_items
from src.io.writer import write_json_file
from src.utils.logging import log_error


def main() -> int:
    """Validate inputs, decode a complete batch, and report failures."""
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
    args = parser.parse_args()

    try:
        settings = RunSettings(
            functions_path=args.functions_definition,
            input_path=args.input,
            output_path=args.output,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            max_context_tokens=args.max_context_tokens,
        )
        prompts = load_prompt_items(settings.input_path)
        functions = load_function_definitions(settings.functions_path)
        if not prompts:
            write_json_file(settings.output_path, [])
            return 0
        if not functions:
            raise ValueError("No available functions for the input prompts.")
        from src.pipeline.function_caller import create_caller

        caller = create_caller(functions, settings)
        calls = caller.run(prompts)
        write_json_file(
            settings.output_path, [call.model_dump() for call in calls]
        )
        return 0
    except KeyboardInterrupt:
        log_error("Interrupted.")
        return 130
    except Exception as error:
        log_error(str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
