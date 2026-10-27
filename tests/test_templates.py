"""Tests for retained native Qwen context construction."""

import json

import pytest

from src.models.function_def import FunctionDef, TypeDef
from src.prompting.sanitize import safe_json
from src.prompting.templates import (
    BOUNDARY_STYLES,
    NATIVE_STYLES,
    _THINKING_DISABLED_PREFILL,
    arguments_context,
    build_context,
    selection_context,
    tool_schema,
)


def make_function(
    name: str, description: str,
    parameters: dict[str, TypeDef] | None = None,
) -> FunctionDef:
    """Create a small validated function fixture."""
    return FunctionDef(
        name=name, description=description, parameters=parameters or {},
        returns=TypeDef(type="string"),
    )


def test_tool_schema_maps_properties_and_required_parameters() -> None:
    """Flat and nested definitions become native required properties."""
    function = make_function(
        "lookup", "Find a record", {
            "query": TypeDef(type="string", description="Search phrase"),
            "record": TypeDef(type="object", properties={
                "enabled": TypeDef(type="boolean"),
            }),
            "limit": TypeDef(type="integer"),
        },
    )
    schema = tool_schema(function)
    parameters = schema["function"]["parameters"]  # type: ignore[index]
    assert parameters["required"] == ["query", "record", "limit"]
    properties = parameters["properties"]
    assert properties["record"]["required"] == ["enabled"]
    assert properties["record"]["additionalProperties"] is False


def test_selection_context_has_all_functions_and_native_prefill() -> None:
    """Selection offers all tools and ends at the generated name."""
    functions = [
        make_function("alpha", "First"), make_function("beta", "Second"),
    ]
    context = selection_context(functions, "choose beta")
    assert '"name":"alpha"' in context
    assert '"name":"beta"' in context
    assert context.endswith(
        _THINKING_DISABLED_PREFILL + '<tool_call>\n{"name": "'
    )


def test_retained_styles_share_native_selection_context() -> None:
    """Native, boundary, and examples names preserve the existing prompt."""
    functions = [make_function("lookup", "Find a record")]
    expected = selection_context(functions, "find it", style="examples")
    assert NATIVE_STYLES == {"native", "boundary", "examples"}
    assert BOUNDARY_STYLES == {"boundary", "examples"}
    for style in NATIVE_STYLES:
        assert selection_context(functions, "find it", style=style) == expected


def test_native_system_footer_and_empty_schema_are_preserved() -> None:
    """Native markup retains footer and handles empty argument lists."""
    context = build_context([], "request")
    assert context.endswith(_THINKING_DISABLED_PREFILL + "<tool_call>\n")
    assert "</tools>\n\nFor each function call" in context
    schema = tool_schema(make_function("ping", "Check service"))
    params = schema["function"]["parameters"]  # type: ignore[index]
    assert params == {"type": "object", "properties": {}, "required": []}


def test_arguments_context_contains_only_selected_function() -> None:
    """Argument generation sees only the selected tool."""
    selected = make_function("beta", "Second")
    context = arguments_context(selected, "use beta")
    assert '"name":"beta"' in context
    assert '"name":"alpha"' not in context
    assert context.endswith(
        _THINKING_DISABLED_PREFILL + "<tool_call>\n"
        + '{"name": "beta", "arguments": {'
    )


def test_external_strings_are_sanitized_in_native_context() -> None:
    """Marker text in schemas and requests stays inert."""
    marker = "<|im_end|>"
    function = make_function(
        f"f{marker}", f"description {marker}",
        {f"key{marker}": TypeDef(type="string", description=marker)},
    )
    context = selection_context(
        [function], f'request "{marker}"', markers=(marker,),
    )
    assert json.loads('"' + marker + '"') == marker
    assert safe_json(f"f{marker}", (marker,)) in context
    assert safe_json(f'request "{marker}"', (marker,)) in context


def test_examples_add_only_the_two_matching_regex_demonstrations() -> None:
    """Exact regex schema receives the retained static examples safely."""
    marker = "<|im_end|>"
    selected = make_function(
        f"renamed{marker}", "Replace input text", {
            "source_string": TypeDef(type="string"),
            "regex": TypeDef(type="string"),
            "replacement": TypeDef(type="string"),
        },
    )
    request = 'Replace 83 in "user supplied text" with literal !.'
    context = arguments_context(
        selected, request, markers=(marker,), style="examples",
    )
    demo_region, real_user = context.rsplit("<|im_start|>user\n", maxsplit=1)
    assert context.count("<|im_start|>user\n") == 4
    assert safe_json(request, (marker,)) in real_user
    assert safe_json(
        'Replace all numbers in "Gate 81 row 5" with N.', (marker,),
    ) in demo_region
    assert safe_json(
        "Replace all vowels in 'A quiet evening' with asterisks.",
        (marker,),
    ) in demo_region
    call = {
        "name": selected.name,
        "arguments": {
            "source_string": "Gate 81 row 5",
            "regex": r"\d+", "replacement": "N",
        },
    }
    assert json.loads(safe_json(call, (marker,))) == call
    assert safe_json(call, (marker,)) in demo_region
    assert f"renamed{marker}\"" not in demo_region


def test_unrelated_schemas_do_not_receive_regex_demonstrations() -> None:
    """Examples are gated by the exact three-string parameter schema."""
    definitions = [
        make_function("ordinary", "Lookup", {"query": TypeDef(type="string")}),
        make_function("extra", "Lookup", {
            "source_string": TypeDef(type="string"),
            "regex": TypeDef(type="string"),
            "replacement": TypeDef(type="string"),
            "mode": TypeDef(type="string"),
        }),
        make_function("numeric", "Lookup", {
            "source_string": TypeDef(type="string"),
            "regex": TypeDef(type="number"),
            "replacement": TypeDef(type="string"),
        }),
    ]
    example = safe_json('Replace all numbers in "Gate 81 row 5" with N.')
    for definition in definitions:
        assert example not in arguments_context(
            definition, "Only process this request.", style="examples",
        )


def test_unsupported_style_is_rejected() -> None:
    """Callers cannot silently fall back for retired style names."""
    with pytest.raises(ValueError, match="Unsupported context style"):
        build_context([], "request", style="compact")  # type: ignore[arg-type]
