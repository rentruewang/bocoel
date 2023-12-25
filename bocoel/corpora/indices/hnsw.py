import numpy as np
from hnswlib import Index as _HnswlibIndex
from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.corpora.interfaces import Embedder, Index, Storage

from . import utils


class HnswlibIndex(Index):
    """
    HNSWLIB index. Uses the hnswlib library.
    """

    def __init__(self, key: str, embeddings: NDArray, threads: int = -1) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings to be 2D, got {embeddings.ndim}D.")

        num_elems, dims = embeddings.shape

        self._key = key
        self._dims = dims
        self._ranges = np.stack([np.min(embeddings), np.max(embeddings)])

        self._index = _HnswlibIndex(max_elements=num_elems, threads=threads)

        embeddings = utils.normalize(embeddings)
        self._index.add_items(embeddings)

    @property
    def key(self) -> str:
        return self._key

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def bounds(self) -> NDArray:
        return self._ranges

    def _search(self, query: NDArray, k: int = 1) -> NDArray:
        return self._index.knn_query(query, k=k)

    @classmethod
    def from_fields(
        cls, store: Storage, emb: Embedder, key: str, threads: int = -1
    ) -> Self:
        items = store.get(key)
        embedded = emb.encode(items)
        return cls(key, embedded, threads=threads)
