from typing import Any

import numpy as np
from numpy import linalg
from numpy.typing import NDArray

from bocoel.corpora.indices import utils
from bocoel.corpora.indices.interfaces import (
    Boundary,
    Distance,
    Index,
    IndexedArray,
    InternalResult,
)


class WhiteningIndex(Index):
    """
    Whitening index. Whitens the data before indexing.
    See https://arxiv.org/abs/2103.15316 for more info.
    """

    def __init__(
        self,
        embeddings: NDArray,
        distance: str | Distance,
        remains: int,
        whitening_backend: type[Index],
        **backend_kwargs: Any,
    ) -> None:
        # Remains might be smaller than embeddings.
        # In such case, no dimensionality reduction is performed.
        remains = min(remains, embeddings.shape[1])

        white = self.whiten(embeddings, remains)
        assert white.shape[1] == remains, {
            "whitened": white.shape,
            "remains": remains,
        }
        self._index = whitening_backend(
            embeddings=white, distance=distance, **backend_kwargs
        )
        assert remains == self._index.dims

    @property
    def batch(self) -> int:
        return self._index.batch

    @property
    def _embeddings(self) -> NDArray | IndexedArray:
        return self._index._embeddings

    @property
    def distance(self) -> Distance:
        return self._index.distance

    @property
    def dims(self) -> int:
        return self._index.dims

    @property
    def boundary(self) -> Boundary:
        return self._index.boundary

    def _search(self, query: NDArray, k: int = 1) -> InternalResult:
        return self._index._search(query, k=k)

    @staticmethod
    def whiten(embeddings: NDArray, k: int) -> NDArray:
        embeddings = utils.normalize(embeddings)

        mean = embeddings.mean(axis=0, keepdims=True)
        covar = np.cov(embeddings.T)

        u, v, _ = linalg.svd(covar)
        sqrt_inv_v = 1 / (v**0.5)
        w = u @ np.diag(sqrt_inv_v)[:, :k]

        white = (embeddings - mean) @ w
        return white
