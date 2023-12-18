from __future__ import annotations

from typing import Container, Mapping, Protocol, Sequence

from corpora.interfaces.embedders import Embedder
from corpora.interfaces.indices import Index
from corpora.interfaces.storages import Storage
from numpy.typing import NDArray


class Corpus(Protocol):
    index: Index
    storage: Storage
    embedder: Embedder

    def keys(self) -> Container:
        return self.storage.keys()

    def __len__(self) -> int:
        return len(self.storage)

    def __getitem__(self, idx: int) -> Mapping[str, str]:
        return self.storage[idx]

    def search(self, query: NDArray, k: int = 1) -> Sequence[Mapping[str, str]]:
        result = self.index(query, k=k)
        items = [self[idx] for idx in result]
        return items
