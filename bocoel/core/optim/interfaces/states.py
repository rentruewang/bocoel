from typing import NamedTuple

from numpy.typing import NDArray

from bocoel.corpora import SearchResult


class State(NamedTuple):
    result: SearchResult
    """
    The search result.
    """

    evaluation: float | NDArray
    """
    The evalution outcome.
    """

    @property
    def query(self) -> NDArray:
        return self.result.query

    @property
    def vectors(self) -> NDArray:
        return self.result.vectors

    @property
    def distances(self) -> NDArray:
        return self.result.distances

    @property
    def indices(self) -> NDArray:
        return self.result.indices
