"""Optional observations emitted after a constrained token is committed."""

from pydantic import BaseModel, ConfigDict
from src.diagnostics.trace_schema import StateSnapshot


class TraceEvent(BaseModel):
    """Immutable token observation; callbacks cannot alter decoder state.

    Attributes:
        step: The 0-based index of the token in the current generation.
        token_id: The selected token ID.
        raw: The selected token's raw bytes.
        allowed_count: The number of allowed candidates at this step.
        complete: Whether the generation is complete after this token.
        state: The stringified constraint state after this token.
        byte_count: The number of bytes generated so far.
        max_steps: The maximum number of steps allowed for this generation.
        before: Optional snapshot of the constraint state before this token.
        after: Optional snapshot of the constraint state after this token.
        elapsed_seconds: Optional time spent in the decoder for this token."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    step: int
    token_id: int
    raw: bytes
    allowed_count: int
    complete: bool
    state: str
    byte_count: int
    max_steps: int
    before: StateSnapshot | None = None
    after: StateSnapshot | None = None
    elapsed_seconds: float | None = None
