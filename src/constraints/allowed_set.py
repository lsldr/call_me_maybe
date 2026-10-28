"""Selection of the highest-scoring token from a constrained set."""

import numpy as np
from numpy.typing import NDArray


def best_allowed(
    logits: NDArray[np.float32], allowed: NDArray[np.int32]
) -> int:
    """Return the highest-scoring allowed ID, with ties favoring lowest ID.

    Args:
        logits: 1D array of scores for all tokens.
        allowed: 1D array of allowed token IDs
                (may be unsorted and have duplicates).

    Returns:
        The ID of the highest-scoring allowed token.

    Raises:
        ValueError: If logits or allowed are not 1D arrays, if allowed contains
            non-integer values, or if allowed is empty."""
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
    return best_sorted_allowed(scores, ids)


def best_sorted_allowed(
    logits: NDArray[np.float32], allowed: NDArray[np.int32]
) -> int:
    """Select from immutable sorted unique IDs without sorting or deduping.

    Args:
        logits: 1D array of scores for all tokens.
        allowed: 1D array of allowed token IDs (sorted and unique).

    Returns:
        The ID of the highest-scoring allowed token.

    Raises:
        ValueError: If logits or allowed are not 1D arrays, if allowed contains
            non-integer values, or if allowed is empty."""
    scores = np.asarray(logits)
    ids = np.asarray(allowed)
    if scores.ndim != 1 or ids.ndim != 1:
        raise ValueError("logits and allowed must be 1D arrays")
    if ids.dtype.kind not in "iu":
        raise ValueError("allowed token IDs must be integers")
    if ids.size == 0:
        raise ValueError("no allowed tokens in this state")
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

    # Caller supplies sorted unique IDs; np.argmax keeps lowest-ID ties.
    best_position = int(np.argmax(allowed_scores))
    return int(ids[best_position])
