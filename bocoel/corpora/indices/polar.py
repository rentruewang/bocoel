from numpy.typing import NDArray

from bocoel.corpora.interfaces import Index, SearchResult

# TODO: Implement polar version of coordinates.


class Polar(Index):
    def _search(self, query: NDArray, k: int = 1) -> SearchResult:
        raise NotImplementedError

    @property
    def bounds(self) -> NDArray:
        raise NotImplementedError

    @property
    def dims(self) -> int:
        raise NotImplementedError
