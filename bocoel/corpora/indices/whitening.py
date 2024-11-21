# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from typing import Any

import numpy as np
import structlog
from numpy import linalg
from numpy.typing import NDArray

from bocoel.corpora.indices import utils
from bocoel.corpora.indices.interfaces import Boundary, Distance, Index, InternalResult

LOGGER = structlog.get_logger()


class WhiteningIndex(Index):
    """
    Whitening index. Whitens the data before indexing.
    See https://arxiv.org/abs/2103.15316 for more info.
    """

    def __init__(
        self,
        embeddings: NDArray,
        distance: str | Distance,
        *,
        reduced: int,
        whitening_backend: type[Index],
        **backend_kwargs: Any,
    ) -> None:
        """
        Initializes the whitening index.

        Parameters:
            embeddings: The embeddings to index.
            distance: The distance metric to use.
            reduced: The reduced dimensionality. NOP if larger than embeddings shape.
            whitening_backend: The backend to use for indexing.
            **backend_kwargs: The backend specific keyword arguments.
        """

        # Reduced might be smaller than embeddings.
        # In such case, no dimensionality reduction is performed.
        if reduced > embeddings.shape[1]:
            reduced = embeddings.shape[1]
            LOGGER.info(
                "Reduced dimensionality is larger than embeddings. Using full dimensionality",
                reduced=reduced,
                embeddings=embeddings.shape,
            )

        white = self.whiten(embeddings, reduced)
        assert white.shape[1] == reduced, {
            "whitened": white.shape,
            "reduced": reduced,
        }
        self._index = whitening_backend(
            embeddings=white, distance=distance, **backend_kwargs
        )
        assert reduced == self._index.dims

    @property
    def batch(self) -> int:
        return self._index.batch

    @property
    def data(self) -> NDArray:
        """
        Returns the data.
        This does not necessarily have the same dimensionality
        as the original transformed embeddings.

        Returns:
            The data.
        """

        return self._index.data

    @property
    def distance(self) -> Distance:
        return self._index.distance

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
