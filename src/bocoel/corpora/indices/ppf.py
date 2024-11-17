# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from collections.abc import Callable
from typing import Any

import numpy as np
import structlog
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from bocoel.common import StrEnum
from bocoel.corpora.indices import utils
from bocoel.corpora.indices.interfaces import Boundary, Distance, Index, InternalResult

LOGGER = structlog.get_logger()


class Distribution(StrEnum):
    """
    The inverse cumulative distribution function (CDF).
    """

    NORMAL = "NORMAL"
    UNIFORM = "UNIFORM"

    @property
    def cdf(self) -> Callable[[ArrayLike], NDArray]:
        """
        Returns the scipy CDF.
        """

        match self:
            case Distribution.NORMAL:
                return stats.norm.cdf
            case Distribution.UNIFORM:
                return stats.uniform.cdf
            case _:
                raise ValueError(f"Unknown CDF: {self}")

    @property
    def ppf(self) -> Callable[[ArrayLike], NDArray]:
        """
        Returns the scipy inverse CDF.
        """

        match self:
            case Distribution.NORMAL:
                return stats.norm.ppf
            case Distribution.UNIFORM:
                return stats.uniform.ppf
            case _:
                raise ValueError(f"Unknown inverse CDF: {self}")


class InverseCDFIndex(Index):
    """
    An index that maps a fixed range [0, 1) with
    the inverse cumulative distribution function (CDF) to index embeddings.
    """

    def __init__(
        self,
        embeddings: NDArray,
        distance: str | Distance,
        *,
        distribution: str | Distribution = Distribution.NORMAL,
        inverse_cdf_backend: type[Index],
        **backend_kwargs: Any,
    ) -> None:
        """
        Parameters:
            embeddings: The embeddings to index.
            distance: The distance metric to use.
            polar_backend: The backend to use for indexing.
            **backend_kwargs: The backend specific keyword arguments.
        """

        embeddings = utils.normalize(embeddings)
        self._index = inverse_cdf_backend(
            embeddings=embeddings,
            distance=distance,
            **backend_kwargs,
        )

        self._distribution = Distribution.lookup(distribution)
        self._data = self._cdf()

    def _search(self, query: NDArray, k: int = 1) -> InternalResult:
        query = self._distribution.ppf(query)
        return self._index._search(query, k=k)

    @property
    def batch(self) -> int:
        return self._index.batch

    @property
    def data(self) -> NDArray:
        return self._data

    @property
    def distance(self) -> Distance:
        return self._index.distance

    @property
    def dims(self) -> int:
        return self._index.dims

    @property
    def boundary(self) -> Boundary:
        EPSILON = 1e-7
        return Boundary.fixed(EPSILON, 1 - EPSILON, dims=self._index.dims)

    def _cdf(self) -> NDArray:
        LOGGER.info(
            "Converting embeddings to polar coordinates.", batch_size=self.batch
        )
        results = []
        for idx in range(len(self._index.data)):
            batch = self._index.data[idx : idx + self.batch]
            results.append(self._distribution.cdf(batch))
        return np.concatenate(results)
