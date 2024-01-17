import abc
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self

from bocoel.common import Batched

from .distances import Distance
from .results import InternalResult, SearchResult


class IndexedArray(Protocol):
    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __getitem__(self, key: int | NDArray, /) -> NDArray:
        ...

    def __array__(self) -> NDArray:
        return np.array([self[idx] for idx in range(len(self))])


class Index(Batched, Protocol):
    """
    Index is responsible for fast retrieval given a vector query.
    """

    def search(self, query: ArrayLike, k: int = 1) -> SearchResult:
        """
        Calls the search function and performs some checks.
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
        vectors = self.embeddings[indices]

        return SearchResult(
            query=query, vectors=vectors, distances=distances, indices=indices
        )

    def in_range(self, query: NDArray) -> bool:
        return all(query >= self.lower[None, :] & query <= self.upper[None, :])

    @property
    def embeddings(self) -> NDArray | IndexedArray:
        """
        The embeddings used by the index.
        """

        ...

    @property
    @abc.abstractmethod
    def distance(self) -> Distance:
        """
        The distance metric used by the index.
        """

        ...

    @property
    @abc.abstractmethod
    def bounds(self) -> NDArray:
        """
        The bounds of the input.

        Returns
        -------

        An ndarray of shape [dims, 2] where the first column is the lower bound,
        and the second column is the upper bound.
        """

        ...

    @property
    @abc.abstractmethod
    def dims(self) -> int:
        """
        The number of dimensions that the query vector should be.
        """

        ...

    @abc.abstractmethod
    def _search(self, query: NDArray, k: int = 1) -> InternalResult:
        """
        Search the index with a given query.

        Parameters
        ----------

        `query: NDArray`
        The query vector. Must be of shape [dims].

        `k: int`
        The number of nearest neighbors to return.

        Returns
        -------

        A numpy array of shape [k].
        This corresponds to the indices of the nearest neighbors.
        """

        ...

    @classmethod
    @abc.abstractmethod
    def from_embeddings(
        cls, embeddings: NDArray, distance: str | Distance, **kwargs: Any
    ) -> Self:
        """
        Constructs a seasrcher from a set of embeddings.

        Parameters
        ----------

        `embeddings: NDArray`
        The embeddings to construct the index from.

        `distance: str | Distance`
        The distance to use. Can be a string or a Distance enum.

        Returns
        -------
        A index.
        """

        ...

    @property
    def lower(self) -> NDArray:
        return self.bounds[:, 0]

    @property
    def upper(self) -> NDArray:
        return self.bounds[:, 1]
