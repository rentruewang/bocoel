from __future__ import annotations

import abc
from typing import Any, Protocol

from numpy.typing import NDArray

from .embedders import Embedder
from .storages import Storage


class Index(Protocol):
    """
    Index is responsible for fast retrieval given a vector query.
    An index can be considered volatile or static,
    as some databases support vector queries natively.
    """

    def __call__(self, query: NDArray, k: int = 1) -> NDArray:
        """
        Calls the search function and performs some checks.
        """

        if (ndim := query.ndim) != 1:
            raise ValueError(
                f"Expected query to be a 1D vector, got a vector of dim {ndim}."
            )

        if (dim := query.shape[0]) != self.dims:
            raise ValueError(f"Expected query to have dimension {self.dims}, got {dim}")

        if k < 1:
            raise ValueError(f"Expected k to be at least 1, got {k}")

        return self.search(query, k=k)

    @abc.abstractmethod
    def key(self) -> str:
        """
        The key in the original table that this index is for.
        """

        ...

    @abc.abstractmethod
    def dims(self) -> int:
        """
        The number of dimensions that the query vector should be.
        """

        ...

    @abc.abstractmethod
    def search(self, query: NDArray, k: int = 1) -> NDArray:
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

    @abc.abstractmethod
    @classmethod
    def from_fields(cls, store: Storage, emb: Embedder, key: str) -> Index:
        """
        Construct an index from fields.

        Parameters
        ----------

        `store: Storage`
        The storage to index.

        `emb: Embedder`
        The embedder to use.

        `key: str`
        The key in the storage to index.

        Returns
        -------

        An index.
        """

        ...
