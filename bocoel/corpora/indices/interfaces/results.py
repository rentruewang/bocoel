from typing import NamedTuple

from numpy.typing import NDArray


class SearchResult(NamedTuple):
    query: NDArray
    """
    Query vector. Should have shape [batch, dims].
    """

    vectors: NDArray
    """
    Nearest neighbors. Should have shape [batch, k, dims].
    """

    distances: NDArray
    """
    Calculated distance. Should have shape [batch, k].
    """

    indices: NDArray
    """
    Index in the original embeddings. Must be integers. Should have shape [batch, k].
    """


class InternalResult(NamedTuple):
    distances: NDArray
    """
    Calculated distance.
    """

    indices: NDArray
    """
    Index in the original embeddings. Must be integers.
    """
