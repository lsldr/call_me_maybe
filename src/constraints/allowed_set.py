"""Selection of the highest-scoring token from a constrained set."""

import numpy as np
from numpy.typing import NDArray


def best_allowed(
    logits: NDArray[np.float32], allowed: NDArray[np.int32]
) -> int:
    """Return the highest-scoring allowed ID, with ties favoring lowest ID."""
    scores = np.asarray(logits)
    ids = np.asarray(allowed)
    if scores.ndim != 1 or ids.ndim != 1:
        raise ValueError("logits and allowed must be 1D arrays")
    if ids.dtype.kind not in "iu":
        raise ValueError("allowed token IDs must be integers")
    if ids.size == 0:
        raise ValueError("no allowed tokens in this state")

    # Sorting and deduplication make tie behavior independent of input order.
    ids = np.unique(ids)
    if np.any(ids < 0) or np.any(ids >= scores.size):
        raise ValueError(
            "allowed token IDs must be in the range [0, logits.size)"
        )

    allowed_scores = scores[ids]
    if np.any(np.isnan(allowed_scores)):
        raise ValueError("logits for allowed tokens contain NaN values")
    if np.any(np.isposinf(allowed_scores)):
        raise ValueError("logits for allowed tokens contain positive infinity")
    finite = np.isfinite(allowed_scores)
    if not np.any(finite):
        raise ValueError("all allowed token logits are -inf")

    # np.argmax returns the first maximum, and ids are sorted ascending.
    best_position = int(np.argmax(allowed_scores))
    return int(ids[best_position])
