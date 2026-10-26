"""Tests for choosing among legal token IDs."""

import numpy as np
import pytest
from numpy.typing import NDArray
from typing import Any, cast

from src.constraints.allowed_set import best_allowed


def test_selects_best_allowed_and_lowest_id_for_ties() -> None:
    scores = np.array([9, 4, 7], dtype=np.float32)
    assert best_allowed(scores, np.array([1, 2], dtype=np.int32)) == 2
    assert best_allowed(
        np.array([5, 5, 1], dtype=np.float32),
        np.array([1, 0, 1], dtype=np.int32),
    ) == 0


@pytest.mark.parametrize("allowed", [np.array([], dtype=np.int32)])
def test_rejects_empty_candidates(allowed: NDArray[np.int32]) -> None:
    with pytest.raises(ValueError, match="no allowed"):
        best_allowed(np.array([1], dtype=np.float32), allowed)


@pytest.mark.parametrize(
    "allowed", [np.array([-1], dtype=np.int32), np.array([2], dtype=np.int32)]
)
def test_rejects_out_of_bounds_ids(allowed: NDArray[np.int32]) -> None:
    with pytest.raises(ValueError, match="range"):
        best_allowed(np.array([1, 2], dtype=np.float32), allowed)


@pytest.mark.parametrize("allowed", [[0.0], [True]])
def test_rejects_non_integer_id_inputs(
    allowed: list[float] | list[bool],
) -> None:
    with pytest.raises(ValueError, match="integers"):
        best_allowed(
            np.array([1], dtype=np.float32),
            cast(NDArray[np.int32], np.asarray(allowed)),
        )


@pytest.mark.parametrize("bad_score", [np.nan, np.inf])
def test_rejects_nan_and_positive_infinity(bad_score: float) -> None:
    with pytest.raises(ValueError):
        best_allowed(
            np.array([bad_score], dtype=np.float32),
            np.array([0], dtype=np.int32),
        )


def test_negative_infinity_is_allowed_when_finite_choice_exists() -> None:
    scores = np.array([-np.inf, 3, -np.inf], dtype=np.float32)
    assert best_allowed(scores, np.array([2, 0, 1], dtype=np.int32)) == 1


def test_rejects_all_negative_infinity() -> None:
    scores = np.array([-np.inf, -np.inf], dtype=np.float32)
    with pytest.raises(ValueError, match="all allowed"):
        best_allowed(scores, np.array([1, 0], dtype=np.int32))


@pytest.mark.parametrize(
    "logits,allowed",
    [
        (np.array([[1]], dtype=np.float32), np.array([0], dtype=np.int32)),
        (np.array([1], dtype=np.float32), np.array([[0]], dtype=np.int32)),
    ],
)
def test_requires_one_dimensional_inputs(
    logits: NDArray[Any], allowed: NDArray[Any]
) -> None:
    with pytest.raises(ValueError, match="1D"):
        best_allowed(
            cast(NDArray[np.float32], logits),
            cast(NDArray[np.int32], allowed),
        )
