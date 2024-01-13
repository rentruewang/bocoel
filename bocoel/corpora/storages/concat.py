from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from bocoel.corpora.storages.interfaces import Storage


# TODO: Add tests.
class ConcatStorage(Storage):
    def __init__(self, storages: Sequence[Storage], /) -> None:
        if len(storages) < 1:
            raise ValueError("At least one storage is required")

        diff_keys = set(set(store.keys()) for store in storages)

        if len(diff_keys) > 1:
            raise ValueError("Keys are not equal")

        # Unpack the only key in `diff_keys`.
        (self._keys,) = diff_keys
        self._storages = storages
        self._prefix_sum = np.cumsum([len(store) for store in self._storages])

    def keys(self) -> Collection[str]:
        return self._keys

    def __len__(self) -> int:
        return self._prefix_sum[-1]

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        if idx < 0:
            idx %= len(self)

        found = np.searchsorted(self._prefix_sum, idx).item()

        return self._storages[found][idx - self._prefix_sum[found]]

    def get(self, key: str) -> Sequence[Any]:
        results: list[Any] = []

        for store in self._storages:
            results.extend(store.get(key))

        return results

    @classmethod
    def join(cls, storages: Iterable[Storage], /) -> Storage:
        storages = list(storages)

        if len(storages) == 1:
            return storages[0]

        return ConcatStorage(storages)
