from __future__ import annotations

import dataclasses as dcls
from typing import AbstractSet, Mapping, Sequence, Type

from numpy.typing import NDArray

from .interfaces.embedders import Embedder
from .interfaces.indices import Index
from .interfaces.storages import Storage


@dcls.dataclass(frozen=True)
class Corpus:
    key: str
    index: Index
    storage: Storage
    embedder: Embedder

    def keys(self) -> AbstractSet:
        return self.storage.keys()

    def __len__(self) -> int:
        return len(self.storage)

    def __getitem__(self, idx: int) -> Mapping[str, str]:
        return self.storage[idx]

    def search(self, query: NDArray, k: int = 1) -> Sequence[Mapping[str, str]]:
        result = self.index(query, k=k)
        items = [self[idx] for idx in result]
        return items

    @classmethod
    def from_fields(
        cls,
        storage: Storage,
        embedder: Embedder,
        key: str,
        idx_cls: Type[Index],
    ) -> Corpus:
        index = idx_cls.from_fields(storage, embedder, key)
        return cls(key, index, storage, embedder)
