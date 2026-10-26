"""Tests for the bounded allowed-token cache."""

import numpy as np
import pytest

from src.constraints.set_cache import AllowedSetCache


def test_lru_recency_counters_and_resident_bytes() -> None:
    """A hit refreshes recency and eviction removes the least recent key."""
    cache = AllowedSetCache(capacity=2)
    first = np.array([1, 2], dtype=np.int32)
    second = np.array([3], dtype=np.int32)
    third = np.array([4, 5, 6], dtype=np.int32)

    cache.put("first", first)
    cache.put("second", second)
    assert cache.entry_count == 2
    assert cache.resident_bytes == first.nbytes + second.nbytes
    assert cache.get("first") is first
    assert cache.get("missing") is None
    cache.put("third", third)

    assert cache.get("second") is None
    assert cache.get("first") is first
    assert cache.get("third") is third
    assert cache.hits == 3
    assert cache.misses == 2
    assert cache.evictions == 1
    assert cache.entry_count == 2
    assert cache.resident_bytes == first.nbytes + third.nbytes


def test_cached_arrays_are_immutable() -> None:
    """Stored candidate arrays cannot be mutated through the returned value."""
    cache = AllowedSetCache()
    candidates = np.array([2, 4], dtype=np.int32)
    cache.put("state", candidates)

    assert candidates.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        candidates[0] = 5


def test_zero_capacity_tracks_misses_without_retaining_arrays() -> None:
    """A disabled cache records misses but retains no data or evictions."""
    cache = AllowedSetCache(capacity=0)
    candidates = np.array([1, 2], dtype=np.int32)
    cache.put("state", candidates)

    assert cache.get("state") is None
    assert cache.hits == 0
    assert cache.misses == 1
    assert cache.evictions == 0
    assert cache.entry_count == 0
    assert cache.resident_bytes == 0


def test_default_capacity_remains_128() -> None:
    """Keep the configured bounded-cache default unchanged."""
    assert AllowedSetCache().capacity == 128
