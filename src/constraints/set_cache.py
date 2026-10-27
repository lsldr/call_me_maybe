"""Bounded, per-vocabulary cache of legal candidate IDs."""

from collections import OrderedDict
from collections.abc import Hashable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


def normalize_allowed(value: NDArray[Any]) -> NDArray[np.int32]:
    """Return owned, immutable, sorted unique int32 candidate IDs."""
    ids = np.asarray(value)
    if ids.ndim != 1:
        raise ValueError("allowed must be a 1D array")
    if ids.dtype.kind not in "iu":
        raise ValueError("allowed token IDs must be integers")
    if ids.size and (
        np.any(ids < 0) or np.any(ids > np.iinfo(np.int32).max)
    ):
        raise ValueError("allowed token IDs must fit in nonnegative int32")
    normalized = np.unique(ids.astype(np.int32, copy=False))
    # Immutable bytes backing prevents callers from resetting writeable=True.
    return np.frombuffer(normalized.tobytes(), dtype=np.int32)


class AllowedSetCache(BaseModel):
    """Retain a bounded number of immutable candidate arrays per decoder."""

    model_config = ConfigDict(extra="forbid")
    capacity: int = Field(default=128, ge=0)
    byte_capacity: int = Field(default=8 * 1024 * 1024, ge=0)
    hits: int = Field(default=0, ge=0)
    misses: int = Field(default=0, ge=0)
    evictions: int = Field(default=0, ge=0)
    oversize_bypasses: int = Field(default=0, ge=0)
    _entries: OrderedDict[Hashable, NDArray[np.int32]] = PrivateAttr(
        default_factory=OrderedDict
    )
    _resident_bytes: int = PrivateAttr(default=0)
    _peak_bytes: int = PrivateAttr(default=0)

    @property
    def entry_count(self) -> int:
        """Return the number of currently cached arrays."""
        return len(self._entries)

    @property
    def resident_bytes(self) -> int:
        """Return the combined NumPy storage used by cached arrays."""
        return self._resident_bytes

    @property
    def peak_bytes(self) -> int:
        """Return the highest resident byte count reached."""
        return self._peak_bytes

    def get(self, key: Hashable) -> NDArray[np.int32] | None:
        """Return cached IDs and mark the entry recently used."""
        value = self._entries.get(key)
        if value is not None:
            self.hits += 1
            self._entries.move_to_end(key)
        else:
            self.misses += 1
        return value

    def put(self, key: Hashable, value: NDArray[Any]) -> NDArray[np.int32]:
        """Normalize and store IDs, evicting least recent entries as needed."""
        normalized = normalize_allowed(value)
        old = self._entries.pop(key, None)
        if old is not None:
            self._resident_bytes -= old.nbytes
        if not self.capacity or not self.byte_capacity:
            return normalized
        if normalized.nbytes > self.byte_capacity:
            self.oversize_bypasses += 1
            return normalized
        while self._entries and (
            len(self._entries) >= self.capacity
            or self._resident_bytes + normalized.nbytes > self.byte_capacity
        ):
            _, removed = self._entries.popitem(last=False)
            self._resident_bytes -= removed.nbytes
            self.evictions += 1
        self._entries[key] = normalized
        self._resident_bytes += normalized.nbytes
        self._peak_bytes = max(self._peak_bytes, self._resident_bytes)
        return normalized
