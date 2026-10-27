"""Small scoped timing counters for named generation workloads."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr


class TimingBucket(BaseModel):
    """Aggregate call, failure, and elapsed-time counts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    calls: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)
    seconds: float = Field(default=0.0, ge=0.0)


class PerformanceStats(BaseModel):
    """Mutable timing aggregates separated by workload and attempt."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
    )

    buckets: dict[str, TimingBucket] = Field(default_factory=dict)
    workload: str = "generation"
    attempt: int = Field(default=1, ge=1)
    _clock: Callable[[], float] = PrivateAttr(default=perf_counter)

    def reset(self) -> None:
        """Discard all accumulated buckets."""
        self.buckets.clear()

    def report(self) -> dict[str, JsonValue]:
        """Return only serializable timing buckets."""
        return {
            name: bucket.model_dump(mode="json")
            for name, bucket in self.buckets.items()
        }


@contextmanager
def measure(
    stats: PerformanceStats | None,
    name: str,
) -> Generator[None]:
    """Record one named operation, including elapsed time on failure."""
    if stats is None:
        yield
        return

    key = f"{stats.workload}.attempt{stats.attempt}.{name}"
    bucket = stats.buckets.setdefault(key, TimingBucket())
    started = stats._clock()
    bucket.calls += 1
    try:
        yield
    except BaseException:
        bucket.failures += 1
        raise
    finally:
        bucket.seconds += max(0.0, stats._clock() - started)
