import dataclasses as dcls
from typing import NamedTuple

from numpy.typing import NDArray


@dcls.dataclass(frozen=True)
class _SearchResult:
    query: NDArray
    """
    Query vector.
    If batched, should have shape [batch, dims].
    Or else, should have shape [dims].
    """

    vectors: NDArray
    """
    Nearest neighbors.
    If batched, should have shape [batch, k, dims].
    Or else, should have shape [k, dims].
    """

    distances: NDArray
    """
    Calculated distance.
    If batched, should have shape [batch, k].
    Or else, should have shape [k].
    """

    indices: NDArray
    """
    Index in the original embeddings. Must be integers.
    If batched, should have shape [batch, k].
    Or else, should have shape [k].
    """


class SearchResultBatch(_SearchResult):
    """
    A batched version of search result.
    """


class SearchResult(_SearchResult):
    """
    A non-batched version of search result.
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
