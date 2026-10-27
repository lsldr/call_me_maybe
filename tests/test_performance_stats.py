"""Model-free checks for scoped performance aggregates."""

import json
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from src.diagnostics.performance import PerformanceStats, TimingBucket, measure


def fake_clock(values: list[float]) -> Callable[[], float]:
    """Return a deterministic clock that fails on unexpected reads."""
    readings = iter(values)
    return lambda: next(readings)


def test_measure_accumulates_calls_and_elapsed_seconds() -> None:
    """Successful measurements aggregate under the active scope."""
    stats = PerformanceStats()
    stats._clock = fake_clock([2.0, 2.25, 3.0, 3.5])

    with measure(stats, "selection"):
        pass
    with measure(stats, "selection"):
        pass

    assert stats.report() == {
        "generation.attempt1.selection": {
            "calls": 2, "failures": 0, "seconds": 0.75,
        },
    }


def test_measure_records_elapsed_time_and_reraises_exception() -> None:
    """Failed calls count and retain time while preserving the exception."""
    stats = PerformanceStats()
    stats._clock = fake_clock([4.0, 4.125])

    with pytest.raises(RuntimeError, match="failed"):
        with measure(stats, "logits"):
            raise RuntimeError("failed")

    assert stats.report()["generation.attempt1.logits"] == {
        "calls": 1, "failures": 1, "seconds": 0.125,
    }


def test_measure_none_does_not_read_clock() -> None:
    """Disabled measurement executes without consulting a clock."""
    with measure(None, "selection"):
        pass


def test_workload_and_attempt_are_captured_at_entry() -> None:
    """Measurements are separated by workload and attempt scope."""
    stats = PerformanceStats(workload="generation", attempt=1)
    stats._clock = fake_clock([0.0, 1.0, 1.0, 3.0])
    with measure(stats, "decode"):
        stats.workload = "retry"
        stats.attempt = 2
    with measure(stats, "decode"):
        pass

    assert stats.report() == {
        "generation.attempt1.decode": {
            "calls": 1, "failures": 0, "seconds": 1.0,
        },
        "retry.attempt2.decode": {
            "calls": 1, "failures": 0, "seconds": 2.0,
        },
    }


def test_reset_clears_buckets_and_report_is_json_safe() -> None:
    """Reset discards aggregates and reports contain no runtime objects."""
    stats = PerformanceStats(buckets={"old": TimingBucket(calls=1)})
    assert json.loads(json.dumps(stats.report())) == {
        "old": {"calls": 1, "failures": 0, "seconds": 0.0},
    }

    stats.reset()

    assert stats.report() == {}


@pytest.mark.parametrize(
    "values",
    [
        {"calls": -1},
        {"failures": -1},
        {"seconds": -0.1},
    ],
)
def test_timing_bucket_rejects_negative_counters(
    values: dict[str, int | float],
) -> None:
    """Timing aggregates cannot contain negative values."""
    with pytest.raises(ValidationError):
        TimingBucket.model_validate(values)
