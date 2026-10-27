"""Validate the handwritten example against the first bundled input."""

from pathlib import Path

from src.io.reader import (
    load_function_definitions,
    load_prompt_items,
    validate_function_calls,
)


def test_first_input_accepts_handwritten_expected_call() -> None:
    """Check exact output shape and scalar types without loading a model."""
    root = Path(__file__).resolve().parents[1]
    functions = load_function_definitions(
        root / "data/input/functions_definition.json"
    )
    prompts = load_prompt_items(
        root / "data/input/function_calling_tests.json"
    )

    expected = [{
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {"a": 2.0, "b": 3.0},
    }]
    calls = validate_function_calls(expected, functions, prompts[:1])

    assert len(calls) == 1
    assert calls[0].model_dump() == expected[0]
    assert type(calls[0].parameters["a"]) is float
    assert type(calls[0].parameters["b"]) is float
