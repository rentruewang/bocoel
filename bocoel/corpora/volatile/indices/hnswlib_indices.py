from __future__ import annotations

import numpy as np
from hnswlib import Index as HnswIdx
from numpy.typing import NDArray

from bocoel.corpora.interfaces import Embedder, Index, Storage

# FIXME: Could raise ValueError instead of using asserts
# FIXME: Threads


class HnswlibIndex(Index):
    def __init__(self, key: str, embeddings: NDArray) -> None:
        assert embeddings.ndim == 2

        num_elems, dims = embeddings.shape

        self.key = key
        self.dims = dims

        self._index = HnswIdx(max_elements=num_elems)
        self._index.add_items(embeddings)

    def search(self, query: NDArray, k: int = 1) -> NDArray:
        return self._index.knn_query(query, k=k)

    @classmethod
    def from_storage(cls, store: Storage, key: str, emb: Embedder) -> HnswlibIndex:
        items = [store[idx][key] for idx in range(len(store))]
        embedded = [emb.encode(text) for text in items]
        assert all(len(e) == emb.dims for e in embedded)

        return cls(key, np.stack(embedded))
