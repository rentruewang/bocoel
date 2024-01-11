from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from bocoel.corpora.storages.interfaces import Storage


# TODO: Add tests.
class ConcatStorage(Storage):
    def __init__(self, storages: Iterable[Storage]) -> None:
        storages = list(storages)

        if len(storages) < 1:
            raise ValueError("At least one storage is required")

        self._storages = storages

        self._keys = self._storages[0].keys()

        for store in self._storages[1:]:
            if self._keys != store.keys():
                raise ValueError("Keys are not equal")

        self._cum_len = np.cumsum([len(store) for store in self._storages])

    def keys(self) -> Collection[str]:
        return self._keys

    def __len__(self) -> int:
        return self._cum_len[-1]

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        if idx < 0:
            idx += len(self)

        found = np.searchsorted(self._cum_len, idx)

        return self._storages[found][idx - self._cum_len[found]]

    def get(self, key: str) -> Sequence[Any]:
        results: list[Any] = []

        for store in self._storages:
            results.extend(store.get(key))

        return results
