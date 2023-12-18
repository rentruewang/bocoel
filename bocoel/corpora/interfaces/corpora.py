from __future__ import annotations

import dataclasses as dcls
from typing import Mapping, Type

from .embedders import Embedder
from .indices import Index
from .storages import Storage


@dcls.dataclass(frozen=True)
class Corpus:
    keys_index: Mapping[str, Index]
    storage: Storage
    embedder: Embedder

    def __len__(self) -> int:
        return len(self.storage)

    def __getitem__(self, idx: int) -> Mapping[str, str]:
        return self.storage[idx]

    @classmethod
    def from_fields(
        cls,
        storage: Storage,
        embedder: Embedder,
        keys_index_factory: Mapping[str, Type[Index]],
    ) -> Corpus:
        keys_index = {
            key: idx_cls.from_fields(storage, embedder, key)
            for key, idx_cls in keys_index_factory.items()
        }
        return cls(storage=storage, embedder=embedder, keys_index=keys_index)
