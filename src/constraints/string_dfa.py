"""A byte-level DFA for JSON strings and their UTF-8 contents."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


StringPhase = Literal[
    "NORMAL", "ESCAPE", "UNICODE", "LOW_SLASH", "LOW_U", "DONE"
]


class StringState(BaseModel):
    """Immutable state for a JSON string prefix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: StringPhase = "NORMAL"
    utf8_remaining: int = 0
    utf8_min: int = 0x80
    utf8_max: int = 0xBF
    hex_remaining: int = 0
    hex_value: int = 0
    low_surrogate: bool = False


class StringDFA(BaseModel):
    """Validate token bytes for one JSON string, including its closing quote.

    The opening quote is supplied before this DFA receives any bytes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def initial_state(self) -> StringState:
        """Return the state immediately after the opening quote."""
        return StringState()

    def advance(self, state: StringState, raw: bytes) -> StringState | None:
        """Consume bytes and return a new state, or ``None`` on rejection."""
        if not raw or state.phase == "DONE":
            return None
        phase: StringPhase = state.phase
        remaining = state.utf8_remaining
        lower, upper = state.utf8_min, state.utf8_max
        hex_remaining, hex_value = state.hex_remaining, state.hex_value
        low_surrogate = state.low_surrogate

        for index, byte in enumerate(raw):
            if phase == "DONE":
                return None
            if remaining:
                if not lower <= byte <= upper:
                    return None
                remaining -= 1
                lower, upper = 0x80, 0xBF
                continue

            if phase == "LOW_SLASH":
                if byte != 0x5C:
                    return None
                phase = "LOW_U"
            elif phase == "LOW_U":
                if byte != ord("u"):
                    return None
                phase, hex_remaining, hex_value = "UNICODE", 4, 0
                low_surrogate = True
            elif phase == "ESCAPE":
                if byte in b'"\\/bfnrt':
                    phase = "NORMAL"
                elif byte == ord("u"):
                    phase, hex_remaining, hex_value = "UNICODE", 4, 0
                    low_surrogate = False
                else:
                    return None
            elif phase == "UNICODE":
                digit = _hex_value(byte)
                if digit is None:
                    return None
                hex_value = (hex_value << 4) | digit
                hex_remaining -= 1
                if hex_remaining == 0:
                    if low_surrogate:
                        if not 0xDC00 <= hex_value <= 0xDFFF:
                            return None
                        phase = "NORMAL"
                        low_surrogate = False
                        hex_value = 0
                    elif 0xD800 <= hex_value <= 0xDBFF:
                        phase = "LOW_SLASH"
                        low_surrogate = False
                        hex_value = 0
                    elif 0xDC00 <= hex_value <= 0xDFFF:
                        return None
                    else:
                        phase = "NORMAL"
                        hex_value = 0
                    hex_remaining = 0
            else:
                if byte < 0x20:
                    return None
                if byte == 0x22:
                    if index != len(raw) - 1:
                        return None
                    phase = "DONE"
                elif byte == 0x5C:
                    phase = "ESCAPE"
                elif byte <= 0x7F:
                    continue
                elif 0xC2 <= byte <= 0xDF:
                    remaining, lower, upper = 1, 0x80, 0xBF
                elif byte == 0xE0:
                    remaining, lower, upper = 2, 0xA0, 0xBF
                elif 0xE1 <= byte <= 0xEC or 0xEE <= byte <= 0xEF:
                    remaining, lower, upper = 2, 0x80, 0xBF
                elif byte == 0xED:
                    remaining, lower, upper = 2, 0x80, 0x9F
                elif byte == 0xF0:
                    remaining, lower, upper = 3, 0x90, 0xBF
                elif 0xF1 <= byte <= 0xF3:
                    remaining, lower, upper = 3, 0x80, 0xBF
                elif byte == 0xF4:
                    remaining, lower, upper = 3, 0x80, 0x8F
                else:
                    return None

        return StringState(
            phase=phase,
            utf8_remaining=remaining,
            utf8_min=lower,
            utf8_max=upper,
            hex_remaining=hex_remaining,
            hex_value=hex_value,
            low_surrogate=low_surrogate,
        )

    def next_bytes(self, state: StringState) -> set[int]:
        """Return every byte that can legally extend ``state``."""
        return {
            byte
            for byte in range(256)
            if self.advance(state, bytes([byte])) is not None
        }

    def is_complete(self, state: StringState) -> bool:
        """Return whether the final unescaped quote has been consumed."""
        return state.phase == "DONE"


def _hex_value(byte: int) -> int | None:
    """Return a hexadecimal digit value, or ``None`` for other bytes."""
    if ord("0") <= byte <= ord("9"):
        return byte - ord("0")
    if ord("a") <= byte <= ord("f"):
        return byte - ord("a") + 10
    if ord("A") <= byte <= ord("F"):
        return byte - ord("A") + 10
    return None
