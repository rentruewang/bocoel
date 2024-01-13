from typing import NamedTuple

from numpy.typing import NDArray


class State(NamedTuple):
    """
    State tracks a single query during evaluation.
    """

    query: NDArray
    """
    The query vector. Must be of shape [dims].
    """

    vectors: NDArray
    """
    Nearest neighbors. Must be of shape [k, dims].
    """

    distances: NDArray
    """
    Calculated distance. Must be of shape [k].
    """

    indices: NDArray
    """
    Index in the original embeddings. Must be integers. Must be of shape [k].
    """

    evaluation: float
    """
    The evalution outcome. Average of the scores of the retrieved vectors.
    """
