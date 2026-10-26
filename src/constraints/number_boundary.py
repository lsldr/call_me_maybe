"""Accept a number delimiter merged with known following syntax."""

from pydantic import BaseModel, ConfigDict, Field

from src.constraints.number_dfa import NumberDFA, NumberState


class NumberBoundaryState(BaseModel):
    """Track the number DFA and any consumed following syntax bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inner: NumberState = Field(default_factory=NumberState)
    suffix: bytes = b""


class NumberBoundaryDFA(BaseModel):
    """Wrap a number DFA to recognize a merged known suffix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    number: NumberDFA = Field(default_factory=NumberDFA)
    following: bytes = b""

    def initial_state(self) -> NumberBoundaryState:
        """Return a wrapper around the number DFA's initial state."""
        return NumberBoundaryState(inner=self.number.initial_state())

    def advance(
        self, state: NumberBoundaryState, raw: bytes,
    ) -> NumberBoundaryState | None:
        """Trial-consume bytes, retaining rather than dropping suffix bytes."""
        if not raw or self.number.is_complete(state.inner):
            return None

        complete_number = self.number.advance(state.inner, raw)
        if complete_number is not None:
            return NumberBoundaryState(inner=complete_number)

        delimiter = ord(self.number.delimiter)
        for index, byte in enumerate(raw):
            if byte != delimiter:
                continue
            number_state = self.number.advance(
                state.inner, raw[:index + 1],
            )
            if number_state is None or not self.number.is_complete(
                number_state
            ):
                continue
            suffix = raw[index + 1:]
            valid_suffix = (
                len(suffix) <= len(self.following)
                and self.following.startswith(suffix)
            )
            if valid_suffix:
                return NumberBoundaryState(
                    inner=number_state, suffix=suffix,
                )
        return None

    def next_bytes(self, state: NumberBoundaryState) -> set[int]:
        """Return legal number bytes until its delimiter is consumed."""
        if self.is_complete(state):
            return set()
        return self.number.next_bytes(state.inner)

    def is_complete(self, state: NumberBoundaryState) -> bool:
        """Delegate completion to the wrapped number DFA."""
        return self.number.is_complete(state.inner)
