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
    first_cached = cache.get("first")
    assert first_cached is not None
    assert cache.get("missing") is None
    cache.put("third", third)

    assert cache.get("second") is None
    assert cache.get("first") is first_cached
    third_cached = cache.get("third")
    assert third_cached is not None
    assert cache.hits == 3
    assert cache.misses == 2
    assert cache.evictions == 1
    assert cache.entry_count == 2
    assert cache.resident_bytes == first.nbytes + third_cached.nbytes


def test_cached_arrays_are_immutable() -> None:
    """Stored candidate arrays cannot be mutated through the returned value."""
    cache = AllowedSetCache()
    candidates = np.array([2, 4], dtype=np.int32)
    cache.put("state", candidates)

    cached = cache.get("state")
    assert cached is not None
    assert cached.flags.writeable is False
    with pytest.raises(ValueError):
        cached.flags.writeable = True
    with pytest.raises(ValueError, match="read-only"):
        cached[0] = 5


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


def test_default_capacities() -> None:
    """Keep the entry limit and initial byte budget bounded."""
    assert AllowedSetCache().capacity == 128
    assert AllowedSetCache().byte_capacity == 8 * 1024 * 1024


def test_put_normalizes_and_owns_input() -> None:
    """External arrays are copied, sorted, deduplicated, and made immutable."""
    cache = AllowedSetCache()
    source = np.array([3, 1, 3], dtype=np.int64)
    normalized = cache.put("key", source)
    source[:] = 0
    assert normalized.tolist() == [1, 3]
    assert cache.get("key") is normalized


def test_byte_capacity_evicts_lru_and_tracks_peak() -> None:
    """Byte pressure evicts least recent entries and updates resident bytes."""
    cache = AllowedSetCache(capacity=10, byte_capacity=12)
    first = cache.put("first", np.array([1, 2], dtype=np.int32))
    cache.put("second", np.array([3], dtype=np.int32))
    assert cache.get("first") is first
    cache.put("third", np.array([4], dtype=np.int32))
    assert cache.get("second") is None
    assert cache.evictions == 1
    assert cache.resident_bytes == 12
    assert cache.peak_bytes == 12


def test_replacement_and_oversize_bypass() -> None:
    """Replacement adjusts bytes; an oversize value is returned uncached."""
    cache = AllowedSetCache(capacity=2, byte_capacity=8)
    cache.put("key", np.array([1], dtype=np.int32))
    replacement = cache.put("key", np.array([2, 3], dtype=np.int32))
    assert replacement.tolist() == [2, 3]
    assert cache.entry_count == 1
    assert cache.resident_bytes == 8
    oversized = cache.put("big", np.array([4, 5, 6], dtype=np.int32))
    assert oversized.tolist() == [4, 5, 6]
    assert cache.get("big") is None
    assert cache.oversize_bypasses == 1
    assert cache.evictions == 0
