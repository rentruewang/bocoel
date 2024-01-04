from typing import NamedTuple

from numpy.typing import NDArray


class SearchResult(NamedTuple):
    query: NDArray
    """
    Query vector.
    """

    vectors: NDArray
    """
    Nearest neighbors.
    """

    distances: NDArray
    """
    Calculated distance.
    """

    indices: NDArray
    """
    Index in the original embeddings. Must be integers.
    """


class InternalSearchResult(NamedTuple):
    distances: NDArray
    """
    Calculated distance.
    """

    indices: NDArray
    """
    Index in the original embeddings. Must be integers.
    """
