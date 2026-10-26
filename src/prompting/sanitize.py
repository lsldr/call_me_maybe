"""Serialize external JSON data while keeping marker spellings inert."""

import json
from collections.abc import Collection
from typing import Any


def _unicode_escape(character: str) -> str:
    """Return a JSON Unicode escape for one Unicode code point."""
    codepoint = ord(character)
    if codepoint <= 0xFFFF:
        return f"\\u{codepoint:04x}"
    codepoint -= 0x10000
    high = 0xD800 + (codepoint >> 10)
    low = 0xDC00 + (codepoint & 0x3FF)
    return f"\\u{high:04x}\\u{low:04x}"


def _safe_string(value: str, markers: tuple[str, ...]) -> str:
    """Encode one string as JSON, escaping template markers in its content."""
    chunks = ['"']
    for index, character in enumerate(value):
        if 0xD800 <= ord(character) <= 0xDFFF:
            raise ValueError(
                "JSON strings cannot contain raw surrogate code points"
            )
        if character == "<" or any(
            "<" not in marker and value.startswith(marker, index)
            for marker in markers
        ):
            chunks.append(_unicode_escape(character))
        elif character == '"':
            chunks.append('\\"')
        elif character == "\\":
            chunks.append("\\\\")
        elif ord(character) < 0x20:
            chunks.append(json.dumps(character, ensure_ascii=False)[1:-1])
        else:
            chunks.append(character)
    chunks.append('"')
    return "".join(chunks)


def _encode(value: Any, markers: tuple[str, ...]) -> str:
    """Recursively encode JSON-compatible data with safe string handling."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _safe_string(value, markers)
    if isinstance(value, (int, float)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_encode(item, markers) for item in value) + "]"
    if isinstance(value, dict):
        items: list[str] = []
        for key, item in value.items():
            if isinstance(key, str):
                key_text = key
            elif key is None:
                key_text = "null"
            elif isinstance(key, bool):
                key_text = "true" if key else "false"
            elif isinstance(key, (int, float)):
                key_text = json.dumps(
                    key, ensure_ascii=False, allow_nan=False
                )
            else:
                raise TypeError(
                    "keys must be str, int, float, bool or None, "
                    f"not {type(key).__name__}"
                )
            encoded_key = _safe_string(key_text, markers)
            items.append(encoded_key + ":" + _encode(item, markers))
        return "{" + ",".join(items) + "}"
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def safe_json(value: object, markers: Collection[str] = ()) -> str:
    """Return compact JSON with angle brackets and external markers escaped.

    The input object is not modified. Empty marker strings are ignored.

    Args:
        value: JSON-compatible data to serialize.
        markers: External token spellings that must not appear literally.

    Returns:
        Compact, Unicode-preserving JSON text that round-trips through
        ``json.loads`` to the original JSON data.
    """
    marker_tuple = tuple(marker for marker in markers if marker)
    if any(not isinstance(marker, str) for marker in marker_tuple):
        raise TypeError("markers must contain only strings")
    return _encode(value, marker_tuple)
