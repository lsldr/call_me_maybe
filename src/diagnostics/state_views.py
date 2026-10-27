"""Explicit parser state display adapters, without repr interpretation."""

import codecs
from collections.abc import Hashable

from src.constraints.container import ContainerState
from src.constraints.number_boundary import NumberBoundaryState
from src.constraints.number_dfa import NumberState
from src.constraints.string_boundary import StringBoundaryState
from src.constraints.string_dfa import StringState
from src.constraints.string_value import StringValueState
from src.diagnostics.trace_schema import StateSnapshot


def snapshot(state: Hashable, fragment: bytes = b"") -> StateSnapshot:
    """Describe known immutable states and any incomplete UTF-8 token bytes.

    Args:
        state: A state already committed by a constraint adapter.
        fragment: This attempt's bytes, bounded by the decoder's byte cap.

    Returns:
        A typed view, or an explicitly unknown state for external adapters.
    """
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        decoder.decode(fragment, final=False)
        pending = decoder.getstate()[0].hex()
    except UnicodeDecodeError:
        pending = ""
    if isinstance(state, StringValueState):
        inner = snapshot(state.inner, fragment)
        return inner.model_copy(update={
            "kind": "string-value", "phase": state.phase + "/" + (
                inner.phase or "unknown"),
        })
    if isinstance(state, StringBoundaryState):
        inner = snapshot(state.inner, fragment)
        return inner.model_copy(update={
            "kind": "string-boundary",
            "details": {**inner.details, "suffix_hex": state.suffix.hex()},
        })
    if isinstance(state, StringState):
        return StateSnapshot(
            kind="string", phase=state.phase, pending_hex=pending,
            details={"utf8_remaining": state.utf8_remaining,
                     "hex_remaining": state.hex_remaining},
        )
    if isinstance(state, NumberBoundaryState):
        return StateSnapshot(kind="number-boundary", phase=state.inner.phase,
                             details={"suffix_hex": state.suffix.hex()})
    if isinstance(state, NumberState):
        return StateSnapshot(kind="number", phase=state.phase)
    if isinstance(state, ContainerState):
        return StateSnapshot(
            kind="container", phase="DONE" if state.done else "CONTENT",
            pending_hex=pending,
            frames=tuple(
                f"node={frame.node} phase={frame.phase} "
                f"index={frame.index} count={frame.count}"
                for frame in state.stack
            ),
            details={"started": state.started,
                     "suffix_hex": state.suffix.hex()},
        )
    if type(state) is int:
        return StateSnapshot(kind="trie", phase="node",
                             details={"node": state})
    return StateSnapshot(kind="unknown")
