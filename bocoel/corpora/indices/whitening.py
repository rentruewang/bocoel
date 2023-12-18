import numpy as np
from numpy import linalg
from numpy.typing import NDArray

from bocoel.corpora.interfaces import Embedder, Index, Storage

from .hnswlib_indices import HnswlibIndex


class WhiteningIndex(Index):
    def __init__(
        self, key: str, embeddings: NDArray, k: int, threads: int = -1
    ) -> None:
        mean = embeddings.mean(axis=0, keepdims=True)
        covar = np.cov(embeddings)

        u, v, _ = linalg.svd(covar)
        sqrt_inv_v = 1 / (v**0.5)
        w = (u @ sqrt_inv_v)[:, :k]

        whitened = (embeddings - mean) @ w

        self._hnswidx = HnswlibIndex(key, whitened, threads)
        assert k == self._hnswidx.dims()

    def key(self) -> str:
        return self._hnswidx.key()

    def dims(self) -> int:
        return self._hnswidx.dims()

    def ranges(self) -> NDArray:
        return self._hnswidx.ranges()

    def search(self, query: NDArray, k: int = 1) -> NDArray:
        return self._hnswidx.search(query, k=k)

    @classmethod
    def from_fields(cls, store: Storage, emb: Embedder, key: str, k: int) -> Index:
        items = [store[idx][key] for idx in range(len(store))]
        embedded = emb(items)
        return cls(key, embedded, k)
