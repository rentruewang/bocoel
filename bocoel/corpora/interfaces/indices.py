import abc
import typing
from typing import NamedTuple, Protocol

from numpy.typing import NDArray


class SearchResult(NamedTuple):
    vectors: NDArray
    distances: NDArray
    indices: NDArray


@typing.runtime_checkable
class Index(Protocol):
    """
    Index is responsible for fast retrieval given a vector query.
    An index can be considered volatile or static,
    as some databases support vector queries natively.
    """

    def search(self, query: NDArray, k: int = 1) -> SearchResult:
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

        return self._search(query, k=k)

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
    def _search(self, query: NDArray, k: int = 1) -> SearchResult:
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

    @property
    def lower(self) -> NDArray:
        return self.bounds[:, 0]

    @property
    def upper(self) -> NDArray:
        return self.bounds[:, 1]
