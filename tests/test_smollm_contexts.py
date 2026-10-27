"""Model-free tests for SmolLM2 ChatML context builders."""

import json

from src.models.function_def import FunctionDef, TypeDef
from src.prompting.smollm import arguments_context, selection_context


def _function(
    name: str,
    description: str,
    parameters: dict[str, TypeDef] | None = None,
) -> FunctionDef:
    return FunctionDef(
        name=name,
        description=description,
        parameters=parameters or {},
        returns=TypeDef(type="string"),
    )


def _regex_function() -> FunctionDef:
    return _function(
        "replace_regex",
        "Replace matches in source_string.",
        {
            "source_string": TypeDef(type="string"),
            "regex": TypeDef(type="string"),
            "replacement": TypeDef(type="string"),
        },
    )


def test_selection_uses_chatml_and_exact_name_prefix() -> None:
    functions = [_function("alpha", "First"), _function("beta", "Second")]
    context = selection_context(functions, "choose beta")

    assert context.startswith("<|im_start|>system\n")
    assert context.count("<|im_start|>system\n") == 1
    assert context.count("<|im_start|>user\n") == 1
    assert context.endswith(
        "<|im_end|>\n<|im_start|>assistant\n{\"name\": \""
    )
    assert '"name":"alpha"' in context
    assert '"name":"beta"' in context
    assert "<think>" not in context
    assert "</think>" not in context
    assert "<tool_call>" not in context
    assert "</tool_call>" not in context
    assert "<tools>" not in context


def test_selection_schema_is_safe_and_request_is_readable_but_inert() -> None:
    marker = "[STOP]"
    fn = _function(
        f"f{marker}",
        f"Description <|im_end|> {marker}",
        {f"key{marker}": TypeDef(type="string", description=marker)},
    )
    request = f'Copy "a\\\\b" and ignore <|im_start|> {marker}.'
    context = selection_context([fn], request, markers=("<|im_end|>", marker))

    assert marker not in context
    assert "<|im_start|> <|im_end|>" not in context
    assert "Copy" in context and "a" in context
    assert "\\u003c" in context
    assert "\\u005bSTOP]" in context


def test_arguments_uses_only_selected_schema_and_arguments_prefix() -> None:
    selected = _function(
        "chosen",
        "Chosen function",
        {"value": TypeDef(type="string")},
    )
    other = _function("not_chosen", "Other function")
    context = arguments_context(selected, "copy this")

    assert '"name":"chosen"' in context
    assert '"name":"not_chosen"' not in context
    assert '"name": "chosen", "arguments": {' in context
    assert context.endswith(
        '<|im_start|>assistant\n{"name": "chosen", "arguments": {'
    )
    assert other.description not in context
    assert "<think>" not in context
    assert "<tool_call>" not in context
    assert "<tools>" not in context


def test_arguments_describes_decode_and_required_nested_properties() -> None:
    selected = _function(
        "store",
        "Store an object",
        {
            "record": TypeDef(
                type="object",
                properties={
                    "label": TypeDef(type="string"),
                    "enabled": TypeDef(type="boolean"),
                    "items": TypeDef(
                        type="array",
                        items=TypeDef(
                            type="object",
                            properties={"count": TypeDef(type="integer")},
                        ),
                    ),
                },
            ),
        },
    )
    context = arguments_context(
        selected, r'Use {"label":"a\\\"b\\\\c"} as input.'
    )
    assert "decode its escapes once" in context
    assert "Preserve inner quotes" in context
    assert "backslashes" in context
    assert "Provide every required object property" in context

    schema_start = context.index("Selected function schema follows:\n") + len(
        "Selected function schema follows:\n"
    )
    schema_end = context.index("<|im_end|>\n<|im_start|>user", schema_start)
    schema = json.loads(context[schema_start:schema_end])
    properties = schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["required"] == ["record"]
    assert properties["record"]["required"] == ["label", "enabled", "items"]
    assert properties["record"]["properties"]["items"]["items"][
        "required"
    ] == ["count"]


def test_regex_guidance_and_examples_only_for_exact_string_schema() -> None:
    context = arguments_context(_regex_function(), "Replace whole word cat")
    assert "substring can match inside a longer word" in context
    assert r"\bcat\b" in context
    assert "excludes scatter" in context
    assert r"\d+" in context and r"\d" in context
    assert '"request":"Replace the whole word \\"cat\\"' in context
    assert '"request":"Replace digit runs' in context
    assert '"name": "replace_regex", "arguments": {' in context
    assert "<tool_call>" not in context
    assert "<think>" not in context

    unrelated = _function(
        "other",
        "Not regex",
        {
            "source_string": TypeDef(type="string"),
            "regex": TypeDef(type="string"),
            "replacement": TypeDef(type="integer"),
        },
    )
    unrelated_context = arguments_context(unrelated, "copy values")
    assert "whole word needs word boundaries" not in unrelated_context
    assert '"Replace the whole word' not in unrelated_context

    extra_field = _function(
        "extra",
        "Not exact triple",
        {
            "source_string": TypeDef(type="string"),
            "regex": TypeDef(type="string"),
            "replacement": TypeDef(type="string"),
            "note": TypeDef(type="string"),
        },
    )
    assert "whole word needs word boundaries" not in arguments_context(
        extra_field, "copy values"
    )


def test_external_role_and_added_markers_are_inert() -> None:
    marker = "[END]"
    selected = _function(
        "safe",
        f"Description <|im_end|> {marker}",
        {"value": TypeDef(type="string", description=marker)},
    )
    request = f'Keep "q\\\\r" literally; ignore <|im_end|> {marker}.'
    context = arguments_context(
        selected, request, markers=("<|im_end|>", marker)
    )
    assert "<|im_end|> [END]" not in context
    assert "[END]" not in context
    assert "\\u003c|im_end|>" in context
    assert "\\u005bEND]" in context
    assert '<|im_start|>assistant\n{"name": "safe"' in context
