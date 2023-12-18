from __future__ import annotations

import abc
from typing import Protocol

from numpy.typing import NDArray

from .embedders import Embedder
from .storages import Storage


class Index(Protocol):
    key: str
    dims: int

    def search(self, query: NDArray, k: int = 1) -> NDArray:
        ...

    @abc.abstractclassmethod
    def from_storage(cls, store: Storage, key: str, emb: Embedder) -> Index:
        ...
