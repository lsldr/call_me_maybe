"""Bounded, per-vocabulary cache of legal candidate IDs."""

from collections import OrderedDict
from collections.abc import Hashable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class AllowedSetCache(BaseModel):
    """Retain a bounded number of immutable candidate arrays per decoder."""

    model_config = ConfigDict(extra="forbid")
    capacity: int = Field(default=128, ge=0)
    hits: int = Field(default=0, ge=0)
    misses: int = Field(default=0, ge=0)
    evictions: int = Field(default=0, ge=0)
    _entries: OrderedDict[Hashable, NDArray[np.int32]] = PrivateAttr(
        default_factory=OrderedDict
    )

    @property
    def entry_count(self) -> int:
        """Return the number of currently cached arrays."""
        return len(self._entries)

    @property
    def resident_bytes(self) -> int:
        """Return the combined NumPy storage used by cached arrays."""
        return sum(value.nbytes for value in self._entries.values())

    def get(self, key: Hashable) -> NDArray[np.int32] | None:
        """Return cached IDs and mark the entry recently used."""
        value = self._entries.get(key)
        if value is not None:
            self.hits += 1
            self._entries.move_to_end(key)
        else:
            self.misses += 1
        return value

    def put(self, key: Hashable, value: NDArray[np.int32]) -> None:
        """Store an immutable array, evicting the least recent entry."""
        if not self.capacity:
            return
        value.flags.writeable = False
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
            self.evictions += 1
