"""Tests for marker-safe JSON serialization."""

import copy
import json

import pytest

from src.prompting.sanitize import safe_json


def test_nested_json_round_trips_and_escapes_markers() -> None:
    """Nested external strings stay intact as data and marker text is inert."""
    value = {
        "<think>": ["before <think> after", {"TOKEN": "TOKEN TOKEN"}],
        "nested": {"<|im_end|>": "literal <|im_end|>"},
    }
    safe = safe_json(value, markers=("<think>", "<|im_end|>", "TOKEN"))
    assert json.loads(safe) == value
    assert "<" not in safe
    assert "TOKEN" not in safe


def test_quotes_backslashes_and_unicode_round_trip_with_markers() -> None:
    """JSON punctuation and non-ASCII marker starts remain valid JSON."""
    value = {"key\"slash\\": 'say "hello" \\ 世界 😀 TOKEN'}
    safe = safe_json(value, markers=('"hello', "\\ 世界", "😀"))
    assert json.loads(safe) == value
    assert '"hello' not in safe
    assert "\\ 世界" not in safe
    assert "😀" not in safe
    assert "世界" in safe


def test_repeated_marker_occurrences_are_all_escaped() -> None:
    """Every occurrence of a plain marker has its first character escaped."""
    safe = safe_json(["TOKEN TOKEN", "xTOKENy"], markers=("TOKEN",))
    assert "TOKEN" not in safe
    assert json.loads(safe) == ["TOKEN TOKEN", "xTOKENy"]


def test_empty_markers_are_ignored_and_input_is_unchanged() -> None:
    """Empty marker spellings are harmless and serialization is nonmutating."""
    value = {"a": ["TOKEN", {"b": 2}]}
    original = copy.deepcopy(value)
    safe = safe_json(value, markers=("", "TOKEN"))
    assert json.loads(safe) == value
    assert value == original


def test_compact_serialization_preserves_unicode() -> None:
    """Compact output keeps ordinary Unicode literal."""
    assert safe_json({"word": "café"}) == '{"word":"café"}'


def test_non_string_keys_match_json_dump_coercion() -> None:
    """Primitive keys use JSON's standard spelling before safe quoting."""
    value = {1: "integer", None: "null", 2.5: "float", False: "boolean"}
    actual = json.loads(safe_json(value))
    expected = json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    )
    assert actual == expected == {
        "1": "integer",
        "null": "null",
        "2.5": "float",
        "false": "boolean",
    }


def test_raw_surrogate_code_points_are_rejected() -> None:
    """Invalid Unicode scalar values cannot leak into UTF-8 JSON text."""
    with pytest.raises(ValueError, match="raw surrogate code points"):
        safe_json("invalid \ud800 text")


def test_nonfinite_numbers_are_rejected() -> None:
    """JSON output never contains NaN or infinity extensions."""
    with pytest.raises(ValueError):
        safe_json(float("nan"))


def test_markers_must_be_strings() -> None:
    """Marker collections reject values outside the string contract."""
    with pytest.raises(TypeError, match="markers must contain only strings"):
        safe_json("data", markers=(1,))  # type: ignore[arg-type]
