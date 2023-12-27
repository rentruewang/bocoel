from typing import Literal

import numpy as np
from hnswlib import Index as _HnswlibIndex
from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.corpora.interfaces import Embedder, Index, SearchResult, Storage

from . import utils

HnswlibDist = Literal["l2", "ip", "cosine"]


class HnswlibIndex(Index):
    """
    HNSWLIB index. Uses the hnswlib library.
    """

    def __init__(
        self, embeddings: NDArray, dist: HnswlibDist, threads: int = -1
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings to be 2D, got {embeddings.ndim}D.")

        embeddings = utils.normalize(embeddings)

        self._emb = embeddings
        self._dist = dist
        self._dims = embeddings.shape[1]
        self._bounds = np.stack([embeddings.min(axis=0), embeddings.max(axis=0)]).T
        assert self._bounds.shape[1] == 2

        # A public attribute because this can be changed at anytime.
        self.threads = threads

        self._init_index()

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def bounds(self) -> NDArray:
        return self._bounds

    def _search(self, query: NDArray, k: int = 1) -> SearchResult:
        indices, distances = self._index.knn_query(query, k=k)
        vectors = self._emb[indices]
        return SearchResult(vectors=vectors, distances=distances, indices=indices)

    def _init_index(self) -> None:
        self._index = _HnswlibIndex(space=self._dist, dim=self.dims)
        self._index.init_index(max_elements=len(self._emb))
        self._index.add_items(self._emb, num_threads=self.threads)

    @classmethod
    def from_fields(
        cls,
        store: Storage,
        emb: Embedder,
        key: str,
        dist: HnswlibDist,
        threads: int = -1,
    ) -> Self:
        items = store.get(key)
        embedded = emb.encode(items)
        return cls(embeddings=embedded, dist=dist, threads=threads)
