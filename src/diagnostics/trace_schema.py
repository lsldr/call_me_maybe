"""Strict schemas for diagnostic trace records."""

from typing import Literal, cast

from pydantic import (
    BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator,
)


class TraceRecord(BaseModel):
    """One versioned JSONL event in a diagnostic run trace."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    version: Literal[1] = 1
    kind: Literal["run_start", "attempt_start",
                  "token", "attempt_end", "run_end"]
    seq: int = Field(ge=0)
    data: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        """Reject boolean/float aliases for the integer format version."""
        if type(value) is not int:
            raise ValueError("trace version must be an integer")
        return value

    @model_validator(mode="after")
    def validate_data(self) -> "TraceRecord":
        """Validate fields whose requirements depend on the event kind.

        Returns:
            This record after successful validation.

        Raises:
            ValueError: If event data is missing or contains invalid fields.
        """
        data = self.data
        if self.kind == "token":
            required = {"step", "token_id", "raw_hex",
                        "allowed_count", "byte_count", "complete"}
            missing = required - data.keys()
            if missing:
                raise ValueError(
                    f"token data missing fields: {', '.join(sorted(missing))}")
            for key in ("step", "token_id"):
                _int_at_least(data, key, 0)
            _int_at_least(data, "allowed_count", 1)
            _int_at_least(data, "byte_count", 0)
            if not isinstance(data["complete"], bool):
                raise ValueError("complete must be a boolean")
            raw_hex = data["raw_hex"]
            if not isinstance(raw_hex, str):
                raise ValueError("raw_hex must be a string")
            try:
                bytes.fromhex(raw_hex)
            except ValueError as exc:
                raise ValueError(
                    "raw_hex must contain valid hexadecimal bytes") from exc
        if self.kind == "attempt_start":
            prompt_index = data.get("prompt_index")
            if ("prompt_index" not in data or
                    (prompt_index is not None and not _is_int(prompt_index))):
                raise ValueError("prompt_index must be an integer or null")
            if _is_int(prompt_index) and cast(int, prompt_index) < 0:
                raise ValueError("prompt_index must be nonnegative")
            if data.get("phase") not in ("selection", "parameter"):
                raise ValueError("phase must be selection or parameter")
            path = data.get("path")
            if not isinstance(path, list) or any(
                    not isinstance(item, (str, int)) or isinstance(item, bool)
                    for item in path):
                raise ValueError("path must be a list of strings or integers")
            _int_at_least(data, "attempt", 1)
            if data["attempt"] not in (1, 2):
                raise ValueError("attempt must be 1 or 2")
            if ("case_id" in data and data["case_id"] is not None
                    and not isinstance(data["case_id"], str)):
                raise ValueError("case_id must be a string or null")
        if "elapsed_seconds" in data:
            elapsed = data["elapsed_seconds"]
            if elapsed is None:
                pass
            elif (not isinstance(elapsed, (float, int))
                  or isinstance(elapsed, bool)):
                raise ValueError(
                    "elapsed_seconds must be finite and nonnegative")
            elif elapsed < 0:
                raise ValueError(
                    "elapsed_seconds must be finite and nonnegative")
            elif (isinstance(elapsed, float) and
                  not (elapsed < float("inf") and elapsed > float("-inf"))):
                raise ValueError("elapsed_seconds must be finite")
        return self


class StateSnapshot(BaseModel):
    """Serializable view of a decoder state without model references."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    kind: str
    phase: str | None = None
    pending_hex: str = ""
    frames: tuple[str, ...] = ()
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("pending_hex")
    @classmethod
    def validate_pending_hex(cls, value: str) -> str:
        """Require the pending decoder bytes to be valid hexadecimal.

        Args:
            value: Hexadecimal text representing incomplete UTF-8 bytes.

        Returns:
            The validated hexadecimal text.

        Raises:
            ValueError: If the value is not valid hexadecimal.
        """
        try:
            bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError(
                "pending_hex must contain hexadecimal bytes") from exc
        return value


def _is_int(value: object) -> bool:
    """Return whether a value is an integer but not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _int_at_least(data: dict[str, JsonValue], key: str, minimum: int) -> None:
    """Require a named JSON field to be an integer at least ``minimum``.

    Args:
        data: Event data mapping to inspect.
        key: Required field name.
        minimum: Inclusive lower bound.

    Raises:
        ValueError: If the field is absent, non-integer, or below the bound.
    """
    value = data.get(key)
    if not _is_int(value) or cast(int, value) < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
