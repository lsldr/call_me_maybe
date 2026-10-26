"""Optional observations emitted after a constrained token is committed."""

from pydantic import BaseModel, ConfigDict


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
