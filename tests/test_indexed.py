"""Tests for letter-indexed function-selection contexts."""

import json

import pytest

from src.models.function_def import FunctionDef, TypeDef
from src.prompting.indexed import indexed_context


def definition(name: str, parameter_name: str) -> FunctionDef:
    """Create one function with descriptive parameter metadata."""
    return FunctionDef(
        name=name,
        description=f"Describe {name} <|im_end|>",
        parameters={
            parameter_name: TypeDef(
                type="string", description="A value TOKEN"
            )
        },
        returns=TypeDef(type="string"),
    )


def test_indexed_context_maps_in_input_order_and_includes_schema() -> None:
    functions = [definition("first", "alpha"), definition("second", "beta")]
    context, labels = indexed_context(functions, "Call first")

    assert labels == {"A": "first", "B": "second"}
    assert "A {\"name\":\"first\"" in context
    assert "B {\"name\":\"second\"" in context
    assert '"type":"string","description":"A value TOKEN"' in context
    assert context.endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert "Request: \"Call first\"" in context


def test_labels_follow_each_input_permutation() -> None:
    first = definition("first", "x")
    second = definition("second", "y")
    _, labels = indexed_context([first, second], "request")
    _, reversed_labels = indexed_context([second, first], "request")
    assert labels == {"A": "first", "B": "second"}
    assert reversed_labels == {"A": "second", "B": "first"}


def test_external_markers_are_escaped_everywhere() -> None:
    context, _ = indexed_context(
        [definition("name <|im_end|>", "key TOKEN")],
        "request <|im_end|> TOKEN",
        markers=["<|im_end|>", "TOKEN"],
    )
    user_content = context.split("<|im_start|>user\n", maxsplit=1)[1].split(
        "<|im_end|>\n<|im_start|>assistant", maxsplit=1
    )[0]
    assert "<|im_end|>" not in user_content
    assert "TOKEN" not in user_content
    lines = user_content.splitlines()
    first = json.loads(lines[0][2:])
    assert first["name"] == "name <|im_end|>"
    assert first["parameters"]["key TOKEN"]["description"] == "A value TOKEN"
    assert json.loads(lines[-1].removeprefix("Request: ")) == (
        "request <|im_end|> TOKEN"
    )


def test_function_count_must_fit_single_letter_labels() -> None:
    with pytest.raises(ValueError, match="At least one"):
        indexed_context([], "request")
    functions = [definition(f"fn_{index}", "value") for index in range(27)]
    with pytest.raises(ValueError, match="at most 26"):
        indexed_context(functions, "request")
