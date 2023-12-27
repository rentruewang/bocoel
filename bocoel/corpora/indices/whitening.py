import numpy as np
from numpy import linalg
from numpy.typing import NDArray

from bocoel.corpora.interfaces import Embedder, Index, SearchResult, Storage

from . import utils
from .hnswlib import HnswlibDist, HnswlibIndex


class WhiteningIndex(Index):
    """
    Whitening index. Uses the hnswlib library but first whitens the data.
    See https://arxiv.org/abs/2103.15316 for more info.
    """

    # FIXME: Maybe I should use *args, **kwargs instead?
    def __init__(
        self, embeddings: NDArray, dist: HnswlibDist, remains: int, threads: int = -1
    ) -> None:
        white = self._whiten(embeddings, remains)
        self._hnswlib_index = HnswlibIndex(embeddings=white, dist=dist, threads=threads)
        assert remains == self._hnswlib_index.dims

    @property
    def dims(self) -> int:
        return self._hnswlib_index.dims

    @property
    def bounds(self) -> NDArray:
        return self._hnswlib_index.bounds

    def _search(self, query: NDArray, k: int = 1) -> SearchResult:
        return self._hnswlib_index.search(query, k=k)

    @classmethod
    def from_fields(
        cls,
        store: Storage,
        emb: Embedder,
        key: str,
        dist: HnswlibDist,
        remains: int,
        threads: int = -1,
    ) -> Index:
        items = store.get(key)
        embedded = emb.encode(items)
        return cls(embeddings=embedded, dist=dist, remains=remains, threads=threads)

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
