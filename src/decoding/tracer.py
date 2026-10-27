"""Optional observations emitted after a constrained token is committed."""

from pydantic import BaseModel, ConfigDict
from src.diagnostics.trace_schema import StateSnapshot


class TraceEvent(BaseModel):
    """Immutable token observation; callbacks cannot alter decoder state."""

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
