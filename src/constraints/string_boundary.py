"""Adapt JSON string validation to tokenizers with merged boundary bytes."""

from pydantic import BaseModel, ConfigDict, Field

from src.constraints.string_dfa import StringDFA, StringState


class StringBoundaryState(BaseModel):
    """Immutable string state and bytes consumed after its closing quote."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inner: StringState = Field(default_factory=StringState)
    suffix: bytes = b""


class StringBoundaryDFA(BaseModel):
    """Validate a JSON string and an optional prefix of following syntax."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    following: bytes

    def initial_state(self) -> StringBoundaryState:
        """Return the state after the string's opening quote."""
        return StringBoundaryState()

    def advance(
        self, state: StringBoundaryState, raw: bytes
    ) -> StringBoundaryState | None:
        """Consume a token without discarding bytes after the string close.

        A token may end within the string, close the string, or continue into
        a prefix of ``following`` after the unescaped closing quote.

        Args:
            state: The current string state and suffix.
            raw: The bytes to consume.

        Returns:
            The new string state and suffix,
            or ``None`` if the token is invalid.
        """
        if not raw or self.is_complete(state):
            return None

        dfa = StringDFA()
        inner_state = dfa.advance(state.inner, raw)
        if inner_state is not None:
            return StringBoundaryState(inner=inner_state)

        for index, byte in enumerate(raw):
            if byte != ord('"'):
                continue
            closed = dfa.advance(state.inner, raw[: index + 1])
            if closed is None or not dfa.is_complete(closed):
                continue
            # The string is closed, so check for a prefix of ``following``.
            extra = raw[index + 1:]
            if len(extra) > len(self.following):
                continue
            if extra == self.following[: len(extra)]:
                return StringBoundaryState(inner=closed, suffix=extra)
        return None

    def next_bytes(self, state: StringBoundaryState) -> set[int]:
        """Return the legal string-content next bytes."""
        if self.is_complete(state):
            return set()
        return StringDFA().next_bytes(state.inner)

    def is_complete(self, state: StringBoundaryState) -> bool:
        """Return whether the string's closing quote has been consumed."""
        return StringDFA().is_complete(state.inner)
