import numpy as np
from numpy import linalg
from numpy.typing import NDArray

from bocoel.corpora.interfaces import Embedder, Index, Storage

from . import utils
from .hnsw import HnswlibIndex


class WhiteningIndex(Index):
    def __init__(
        self, key: str, embeddings: NDArray, k: int, threads: int = -1
    ) -> None:
        white = self._whiten(embeddings, k)
        self._hnswidx = HnswlibIndex(key, white, threads)
        assert k == self._hnswidx.dims

    @property
    def key(self) -> str:
        return self._hnswidx.key

    @property
    def dims(self) -> int:
        return self._hnswidx.dims

    @property
    def bounds(self) -> NDArray:
        return self._hnswidx.bounds

    def _search(self, query: NDArray, k: int = 1) -> NDArray:
        return self._hnswidx.search(query, k=k)

    @classmethod
    def from_fields(
        cls, store: Storage, emb: Embedder, key: str, k: int, threads: int = -1
    ) -> Index:
        items = store.get(key)
        embedded = emb.encode(items)
        return cls(key, embedded, k, threads=threads)

    @staticmethod
    def _whiten(embeddings: NDArray, k: int) -> NDArray:
        embeddings = utils.normalize(embeddings)

        mean = embeddings.mean(axis=0, keepdims=True)
        covar = np.cov(embeddings)

        u, v, _ = linalg.svd(covar)
        sqrt_inv_v = 1 / (v**0.5)
        w = (u @ sqrt_inv_v)[:, :k]

        white = (embeddings - mean) @ w
        return white
