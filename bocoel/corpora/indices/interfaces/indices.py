import abc
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bocoel import common

from .boundaries import Boundary
from .distances import Distance
from .results import InternalResult, SearchResultBatch


class Index(Protocol):
    """
    Index is responsible for fast retrieval given a vector query.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Included s.t. constructors of Index can be used.
        ...

    def __repr__(self) -> str:
        name = common.remove_base_suffix(self, Index)
        return f"{name}({self.dims})"

    def search(self, query: ArrayLike, k: int = 1) -> SearchResultBatch:
        """
        Calls the search function and performs some checks.

        Parameters:
            query: The query vector. Must be of shape `[batch, dims]`.
            k: The number of nearest neighbors to return.

        Returns:
            A `SearchResultBatch` instance. See `SearchResultBatch` for details.
        """

        query = np.array(query)

        if (ndim := query.ndim) != 2:
            raise ValueError(
                f"Expected query to be a 2D vector, got a vector of dim {ndim}."
            )

        if (dim := query.shape[1]) != self.dims:
            raise ValueError(f"Expected query to have dimension {self.dims}, got {dim}")

        if k < 1:
            raise ValueError(f"Expected k to be at least 1, got {k}")

        results: list[InternalResult] = []
        for idx in range(0, len(query), self.batch):
            query_batch = query[idx : idx + self.batch]
            result = self._search(query_batch, k=k)
            results.append(result)

        indices = np.concatenate([res.indices for res in results], axis=0)
        distances = np.concatenate([res.distances for res in results], axis=0)
        vectors = self.data[indices]

        return SearchResultBatch(
            query=query, vectors=vectors, distances=distances, indices=indices
        )

    def in_range(self, query: NDArray) -> bool:
        return all(query >= self.lower[None, :] & query <= self.upper[None, :])

    @property
    @abc.abstractmethod
    def data(self) -> NDArray:
        """
        The underly data that the index is used for searching.
        """

        ...

    @property
    @abc.abstractmethod
    def batch(self) -> int:
        """
        The batch size used for searching.
        """

        ...

    @property
    @abc.abstractmethod
    def boundary(self) -> Boundary:
        """
        The boundary of the input.
        """

        ...

    @property
    @abc.abstractmethod
    def distance(self) -> Distance:
        """
        The distance metric used by the index.
        """

        ...

    @abc.abstractmethod
    def _search(self, query: NDArray, k: int = 1) -> InternalResult:
        """
        Search the index with a given query.

        Parameters:
            query: The query vector. Must be of shape [dims].
            k: The number of nearest neighbors to return.

        Returns:
            A numpy array of shape [k].
                This corresponds to the indices of the nearest neighbors.
        """

        ...

    @property
    def dims(self) -> int:
        """
        The number of dimensions that the query vector should be.
        """

        return self.boundary.dims

    @property
    def lower(self) -> NDArray:
        return self.boundary.lower

    @property
    def upper(self) -> NDArray:
        return self.boundary.upper
