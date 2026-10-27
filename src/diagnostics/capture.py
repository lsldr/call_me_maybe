"""Attach bounded, observational trace capture to a function caller."""

from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import JsonValue

from src.config import RunSettings
from src.decoding.tracer import TraceEvent
from src.diagnostics.trace_io import TraceWriter
from src.llm.profiles import get_profile

if TYPE_CHECKING:
    from src.pipeline.function_caller import FunctionCaller


def start_capture(writer: TraceWriter, settings: RunSettings) -> None:
    """Record actual configuration and source provenance before inference."""
    profile = get_profile(settings.model_name)
    writer.emit("run_start", {
        "run_id": uuid4().hex,
        "model": profile.model_id, "revision": profile.revision,
        "revision_enforced": settings.profile == "bonus",
        "tokenizer": settings.tokenizer, "profile": settings.profile,
        "settings": settings.model_dump(mode="json"),
        "provenance": "live_capture",
    })


def attach_capture(caller: "FunctionCaller", writer: TraceWriter) -> None:
    """Install observers without inspecting model internals or re-scoring."""
    previous_trace = caller.decoder.trace_observer
    previous_attempt = caller.attempt_observer

    def token(event: TraceEvent) -> None:
        data: dict[str, JsonValue] = {
            "step": event.step, "token_id": event.token_id,
            "raw_hex": event.raw.hex(), "allowed_count": event.allowed_count,
            "byte_count": event.byte_count, "complete": event.complete,
            "max_steps": event.max_steps,
            "max_value_bytes": caller.decoder.max_value_bytes,
            "elapsed_seconds": event.elapsed_seconds,
            "typed_state_before": event.before.model_dump(mode="json")
            if event.before is not None else None,
            "typed_state_after": event.after.model_dump(mode="json")
            if event.after is not None else None,
        }
        writer.emit("token", data)
        if previous_trace is not None:
            previous_trace(event)

    def attempt(kind: str, data: dict[str, JsonValue]) -> None:
        writer.emit(kind, data)
        if previous_attempt is not None:
            previous_attempt(kind, data)

    caller.decoder.trace_observer = token
    caller.attempt_observer = attempt
