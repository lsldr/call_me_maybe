"""Tests for prompt context and native tool schema construction."""

import json

from src.models.function_def import FunctionDef, TypeDef
from src.prompting.templates import (
    _THINKING_DISABLED_PREFILL,
    arguments_context,
    build_context,
    selection_context,
    string_argument_context,
    tool_schema,
)
from src.prompting.sanitize import safe_json


def make_function(
    name: str,
    description: str,
    parameters: dict[str, TypeDef] | None = None,
) -> FunctionDef:
    """Create a small validated function fixture."""
    return FunctionDef(
        name=name,
        description=description,
        parameters=parameters or {},
        returns=TypeDef(type="string"),
    )


def test_tool_schema_maps_properties_and_required_parameters() -> None:
    """Flat parameter definitions become native properties and required."""
    function = make_function(
        "lookup",
        "Find a record",
        {
            "query": TypeDef(type="string", description="Search phrase"),
            "limit": TypeDef(type="integer"),
        },
    )
    schema = tool_schema(function)
    assert schema == {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Find a record",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search phrase",
                    },
                    "limit": {"type": "integer"},
                },
                "required": ["query", "limit"],
            },
        },
    }


def test_empty_parameters_produce_empty_required_and_properties() -> None:
    """Functions with no arguments still have a valid object schema."""
    schema = tool_schema(make_function("ping", "Check service"))
    params = schema["function"]["parameters"]  # type: ignore[index]
    assert params == {"type": "object", "properties": {}, "required": []}


def test_selection_context_has_all_functions_and_native_prefill() -> None:
    """Phase one offers all tools and ends at the start of the name value."""
    functions = [
        make_function("alpha", "First"),
        make_function("beta", "Second"),
    ]
    context = selection_context(functions, "choose beta")
    assert '"name":"alpha"' in context
    assert '"name":"beta"' in context
    assert context.endswith(
        _THINKING_DISABLED_PREFILL + '<tool_call>\n{"name": "'
    )
    assert "<tool_call>\n" in context


def test_native_system_footer_matches_recorded_tool_call_example() -> None:
    """The native tool instructions include the recorded call example."""
    context = build_context([], "request")
    system = context.split("<|im_start|>user\n", maxsplit=1)[0]
    assert system.endswith(
        "</tools>\n\nFor each function call, return a json object with "
        "function name and arguments within "
        "<tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call><|im_end|>\n"
    )


def test_arguments_context_contains_only_selected_function() -> None:
    """Phase two keeps only the chosen tool and starts its argument object."""
    beta = make_function("beta", "Second")
    context = arguments_context(beta, "use beta")
    assert '"name":"beta"' in context
    assert '"name":"alpha"' not in context
    assert context.endswith(
        _THINKING_DISABLED_PREFILL
        + "<tool_call>\n"
        + '{"name": "beta", "arguments": {'
    )


def test_compact_style_has_json_definitions_and_matching_prefill() -> None:
    """Compact prompts list flat definitions and omit native tool wrappers."""
    function = make_function("compact_fn", "Description")
    context = build_context([function], "request", style="compact")
    assert "Functions:\n[{" in context
    assert "<tool_call>" not in context
    assert context.endswith(_THINKING_DISABLED_PREFILL)
    assert selection_context([function], "request", style="compact").endswith(
        _THINKING_DISABLED_PREFILL + '{"name": "'
    )


def test_external_strings_and_keys_are_sanitized_in_both_styles() -> None:
    """External schema strings and requests cannot add marker tokens."""
    marker = "<|special_false|>"
    function = make_function(
        f"f{marker}",
        f"description {marker}",
        {f"key{marker}": TypeDef(type="string", description=marker)},
    )
    for style in ("native", "compact"):
        context = selection_context(
            [function],
            f'request "{marker}"',
            markers=(marker,),
            style=style,
        )
        assert marker not in context
        assert json.loads('"' + marker + '"') == marker
        assert "\\u003c" in context


def test_build_context_keeps_request_json_encoded_and_unicode() -> None:
    """The request stays JSON encoded and preserves Unicode."""
    request = 'say "hi" about café'
    context = build_context([], request, style="compact")
    assert '"say \\"hi\\" about café"' in context


def test_string_argument_context_carries_sanitized_json_data() -> None:
    """Focused string prompts preserve selected metadata and prior values."""
    marker = "<|im_end|>"
    selected = make_function(
        "lookup",
        f"Find {marker} records",
        {"query": TypeDef(type="string", description=f"Search {marker}")},
    )
    request = f"Find {marker} for Ada"
    previous: dict[str, object] = {"prior": f"value {marker} TOKEN"}
    context = string_argument_context(
        selected,
        request,
        "query",
        previous,
        markers=(marker, "TOKEN"),
    )
    assert context.endswith(_THINKING_DISABLED_PREFILL + '"')
    assert context.count("<|im_start|>assistant") == 1
    user_data = context.split("<|im_start|>user\n", maxsplit=1)[1].split(
        "<|im_end|>\n<|im_start|>assistant", maxsplit=1
    )[0]
    assert marker not in user_data
    assert "TOKEN" not in user_data
    assert json.loads(user_data) == {
        "function": selected.description,
        "request": request,
        "argument": "query",
        "description": selected.parameters["query"].description,
        "previous_arguments": previous,
    }


def test_examples_style_inserts_safe_matching_schema_demos() -> None:
    """Matching regex schemas get static safe examples with selected name."""
    marker = "<|im_end|>"
    selected = make_function(
        f"renamed{marker}",
        "Replace input text",
        {
            "source_string": TypeDef(type="string"),
            "regex": TypeDef(type="string"),
            "replacement": TypeDef(type="string"),
        },
    )
    actual_request = 'Replace 83 in "user supplied text" with literal !.'
    markers = (marker,)
    context = arguments_context(
        selected, actual_request, markers=markers, style="examples"
    )
    real_user = context.rfind("<|im_start|>user\n")
    demo_region = context[:real_user]

    assert context.count("<|im_start|>user\n") == 3
    assert safe_json(actual_request, markers) in context[real_user:]
    assert actual_request not in demo_region
    demo_request = safe_json(
        "Replace all numbers in 'Gate 81 row 5' with N.", markers
    )
    assert demo_request in demo_region
    assert safe_json(
        "Replace all vowels in 'A quiet evening' with asterisks.", markers
    ) in demo_region

    demo_call = {
        "name": selected.name,
        "arguments": {
            "source_string": "Gate 81 row 5",
            "regex": r"\d+",
            "replacement": "N",
        },
    }
    encoded_call = safe_json(demo_call, markers)
    assert encoded_call in demo_region
    assert json.loads(encoded_call) == demo_call
    assert f'renamed{marker}"' not in demo_region
    assert r"\u003c" in context
    assert '"name":"renamed' in demo_region


def test_examples_style_requires_matching_string_regex_schema() -> None:
    """Other parameter schemas do not receive unrelated demonstrations."""
    non_regex = make_function(
        "ordinary",
        "Lookup",
        {"query": TypeDef(type="string")},
    )
    extra_key = make_function(
        "with_extra",
        "Lookup",
        {
            "source_string": TypeDef(type="string"),
            "regex": TypeDef(type="string"),
            "replacement": TypeDef(type="string"),
            "mode": TypeDef(type="string"),
        },
    )
    example_request = safe_json(
        "Replace all numbers in 'Gate 81 row 5' with N."
    )
    wrong_type = make_function(
        "numeric_pattern", "Lookup",
        {
            "source_string": TypeDef(type="string"),
            "regex": TypeDef(type="number"),
            "replacement": TypeDef(type="string"),
        },
    )
    for selected in (non_regex, extra_key, wrong_type):
        context = arguments_context(
            selected, "Only process this request.", style="examples"
        )
        assert example_request not in context
