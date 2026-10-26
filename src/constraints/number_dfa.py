"""A small DFA for the decimal subset used by generated JSON numbers."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


NumberPhase = Literal[
    "START", "SPACE", "SIGN", "ZERO", "INT", "DOT", "FRAC", "DONE"
]


class NumberState(BaseModel):
    """Immutable state describing the current numeric prefix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: NumberPhase = "START"


class NumberDFA(BaseModel):
    """Validate byte sequences that extend a JSON decimal number."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    integer: bool = False
    delimiter: Literal[",", "}"] = "}"
    allow_leading_space: bool = False

    def initial_state(self) -> NumberState:
        """Return the state before any number bytes have been consumed."""
        return NumberState()

    def advance(self, state: NumberState, raw: bytes) -> NumberState | None:
        """Trial-consume ``raw`` and return its resulting state, or reject it.

        The input state is never changed. The expected structural delimiter is
        accepted only after a complete number and only as the final byte.
        """
        if not raw or state.phase == "DONE":
            return None

        phase = state.phase
        for index, byte in enumerate(raw):
            if byte == ord(self.delimiter):
                if index != len(raw) - 1 or phase not in {
                    "ZERO", "INT", "FRAC"
                }:
                    return None
                return NumberState(phase="DONE")

            digit = ord("0") <= byte <= ord("9")
            if phase == "START":
                if byte == ord(" ") and self.allow_leading_space:
                    phase = "SPACE"
                elif byte == ord("-"):
                    phase = "SIGN"
                elif byte == ord("0"):
                    phase = "ZERO"
                elif ord("1") <= byte <= ord("9"):
                    phase = "INT"
                else:
                    return None
            elif phase == "SPACE":
                if byte == ord("-"):
                    phase = "SIGN"
                elif byte == ord("0"):
                    phase = "ZERO"
                elif ord("1") <= byte <= ord("9"):
                    phase = "INT"
                else:
                    return None
            elif phase == "SIGN":
                if byte == ord("0"):
                    phase = "ZERO"
                elif ord("1") <= byte <= ord("9"):
                    phase = "INT"
                else:
                    return None
            elif phase == "ZERO":
                if byte == ord(".") and not self.integer:
                    phase = "DOT"
                else:
                    return None
            elif phase == "INT":
                if digit:
                    phase = "INT"
                elif byte == ord(".") and not self.integer:
                    phase = "DOT"
                else:
                    return None
            elif phase == "DOT":
                if digit:
                    phase = "FRAC"
                else:
                    return None
            elif phase == "FRAC":
                if not digit:
                    return None
            else:
                return None

        return NumberState(phase=phase)

    def next_bytes(self, state: NumberState) -> set[int]:
        """Return byte values legal as the next byte in ``state``."""
        digits = set(range(ord("0"), ord("9") + 1))
        if state.phase == "START":
            result = {ord("-"), ord("0"), *range(ord("1"), ord("9") + 1)}
            if self.allow_leading_space:
                result.add(ord(" "))
            return result
        if state.phase == "SPACE":
            return {ord("-"), ord("0"), *range(ord("1"), ord("9") + 1)}
        if state.phase == "SIGN":
            return {ord("0"), *range(ord("1"), ord("9") + 1)}
        if state.phase == "ZERO":
            result = {ord(self.delimiter)}
            if not self.integer:
                result.add(ord("."))
            return result
        if state.phase == "INT":
            result = digits | {ord(self.delimiter)}
            if not self.integer:
                result.add(ord("."))
            return result
        if state.phase == "DOT":
            return digits
        if state.phase == "FRAC":
            return digits | {ord(self.delimiter)}
        return set()

    def is_complete(self, state: NumberState) -> bool:
        """Return whether a delimiter has ended the number token."""
        return state.phase == "DONE"
