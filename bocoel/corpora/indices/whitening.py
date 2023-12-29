from typing import Any

import numpy as np
from numpy import linalg
from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.corpora.indices import utils
from bocoel.corpora.interfaces import Distance, Index, SearchResult


class WhiteningIndex(Index):
    """
    Whitening index. Uses the hnswlib library but first whitens the data.
    See https://arxiv.org/abs/2103.15316 for more info.
    """

    def __init__(
        self,
        embeddings: NDArray,
        remains: int,
        distance: Distance,
        idx_cls: type[Index],
        **kwargs: Any
    ) -> None:
        # Remains might be smaller than embeddings.
        # In such case, no dimensionality reduction is performed.
        remains = min(remains, embeddings.shape[1])

        white = self._whiten(embeddings, remains)
        assert white.shape[1] == remains, {
            "whitened": white.shape,
            "remains": remains,
        }
        self._index = idx_cls.from_embeddings(
            embeddings=white, distance=distance, **kwargs
        )
        assert remains == self._index.dims

    @property
    def distance(self) -> Distance:
        return self._index.distance

    @property
    def dims(self) -> int:
        return self._index.dims

    @property
    def bounds(self) -> NDArray:
        return self._index.bounds

    def _search(self, query: NDArray, k: int = 1) -> SearchResult:
        return self._index.search(query, k=k)

    @classmethod
    def from_embeddings(
        cls, embeddings: NDArray, distance: str | Distance, **kwargs: Any
    ) -> Self:
        raise NotImplementedError

    @staticmethod
    def _whiten(embeddings: NDArray, k: int) -> NDArray:
        embeddings = utils.normalize(embeddings)

        mean = embeddings.mean(axis=0, keepdims=True)
        covar = np.cov(embeddings.T)

        u, v, _ = linalg.svd(covar)
        sqrt_inv_v = 1 / (v**0.5)
        w = u @ np.diag(sqrt_inv_v)[:, :k]

        white = (embeddings - mean) @ w
        return white
