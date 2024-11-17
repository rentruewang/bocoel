# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import structlog

from bocoel.corpora.storages.interfaces import Storage

LOGGER = structlog.get_logger()


class ConcatStorage(Storage):
    """
    Storage that concatenates multiple storages together.
    Concatenation is done on the first dimension.
    The resulting storage is read-only and has length
    equal to the sum of the lengths of the storages.
    """

    def __init__(self, storages: Sequence[Storage], /) -> None:
        if len(storages) < 1:
            raise ValueError("At least one storage is required")

        diff_keys = set(frozenset(store.keys()) for store in storages)
        if len(diff_keys) > 1:
            raise ValueError("Keys are not equal")

        # Unpack the only key in `diff_keys`.
        (self._keys,) = diff_keys
        self._storages = tuple(storages)

        LOGGER.info("Concat storage created", storages=storages, keys=diff_keys)

        storage_lengths = [len(store) for store in self._storages]
        self._prefix_sum = np.cumsum(storage_lengths).tolist()
        self._length = sum(storage_lengths)

    def __repr___(self) -> str:
        return f"Concat({list(self._storages)})"

    def keys(self) -> Collection[str]:
        return self._keys

    def __len__(self) -> int:
        return self._length

    def _getitem(self, idx: int) -> Mapping[str, Any]:
        if not -len(self) <= idx < len(self):
            raise IndexError(
                f"Index {idx} is out of bounds. Storage length is {len(self)}"
            )

        if idx < 0:
            idx %= len(self)

        found = np.searchsorted(self._prefix_sum, idx).item()
        sub_idx = idx - self._prefix_sum[found]
        assert sub_idx <= 0, {
            "sub_idx": sub_idx,
            "idx": idx,
            "found": found,
            "prefix_sum": self._prefix_sum,
        }

        return self._storages[found][sub_idx]

    @classmethod
    def join(cls, storages: Iterable[Storage], /) -> Storage:
        storages = list(storages)

        if len(storages) == 1:
            return storages[0]

        return ConcatStorage(storages)
