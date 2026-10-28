"""Recognize a JSON string value including opening quote and boundary bytes."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.constraints.string_boundary import (
    StringBoundaryDFA,
    StringBoundaryState,
)

StringValuePhase = Literal["START", "SPACE", "CONTENT"]


class StringValueState(BaseModel):
    """Immutable prefix state for a full JSON string value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: StringValuePhase = "START"
    inner: StringBoundaryState = Field(default_factory=StringBoundaryState)


class StringValueDFA(BaseModel):
    """Validate optional leading space, both quotes, content, and boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    following: bytes

    def initial_state(self) -> StringValueState:
        """Return the state before an optional space or opening quote."""
        return StringValueState()

    def advance(
        self, state: StringValueState, raw: bytes
    ) -> StringValueState | None:
        """Consume a token while preserving the full string value spelling.

        Args:
            state: The current prefix state.
            raw: The bytes to consume.

        Returns:
            The new prefix state, or ``None`` if the token is invalid."""
        if not raw or self.is_complete(state):
            return None

        phase = state.phase
        inner = state.inner
        index = 0
        if phase == "START":
            if raw[0] == ord(" "):
                phase = "SPACE"
                index = 1
                # Skip any additional spaces.
                if index == len(raw):
                    return StringValueState(phase=phase, inner=inner)
            elif raw[0] == ord('"'):
                phase = "CONTENT"
                index = 1
            else:
                return None

        if phase == "SPACE":
            if raw[index] != ord('"'):
                return None
            phase = "CONTENT"
            index += 1

        if phase == "CONTENT" and index < len(raw):
            boundary = StringBoundaryDFA(following=self.following)
            next_inner = boundary.advance(inner, raw[index:])
            if next_inner is None:
                return None
            inner = next_inner

        return StringValueState(phase=phase, inner=inner)

    def next_bytes(self, state: StringValueState) -> set[int]:
        """Return legal next bytes for this full-value prefix.

        Args:
            state: The current prefix state.

        Returns:
            A set of legal next bytes."""
        if self.is_complete(state):
            return set()
        if state.phase == "START":
            return {ord(" "), ord('"')}
        if state.phase == "SPACE":
            return {ord('"')}
        return StringBoundaryDFA(following=self.following).next_bytes(
            state.inner
        )

    def is_complete(self, state: StringValueState) -> bool:
        """Return whether the string's close and following prefix were read.

        Args:
            state: The current prefix state.

        Returns:
            True if the string is complete, False otherwise."""
        return state.phase == "CONTENT" and StringBoundaryDFA(
            following=self.following
        ).is_complete(state.inner)
