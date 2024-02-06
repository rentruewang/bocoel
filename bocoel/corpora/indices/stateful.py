from collections.abc import Mapping, Sequence

from numpy.typing import ArrayLike, NDArray

from bocoel.corpora.indices import utils
from bocoel.corpora.indices.interfaces import Boundary, Distance, Index, SearchResult
from bocoel.corpora.indices.interfaces.results import InternalResult


class StatefulIndex(Index):
    """
    An index that tracks states.
    """

    def __init__(self, index: Index) -> None:
        self._index = index
        self._clear_history()

    def __repr__(self) -> str:
        return f"Stateful({self._index})"

    def __len__(self) -> int:
        return len(self._history)

    def __getitem__(self, key: int, /) -> SearchResult:
        return self._history[key]

    def _search(self, query: NDArray, k: int = 1) -> InternalResult:
        # This search method doesn't track states.
        return self._index._search(query=query, k=k)

    @property
    def batch(self) -> int:
        return self._index.batch

    @property
    def data(self) -> NDArray:
        return self._index.data

    @property
    def distance(self) -> Distance:
        return self._index.distance

    def stateful_search(
        self, query: ArrayLike, k: int = 1
    ) -> Mapping[int, SearchResult]:
        """
        Search while tracking states.

        Parameters:
            query: The query vector. Must be of shape [dims].
            k: The number of nearest neighbors to return.

        Returns:
            A mapping from the index of the search to the search result.
            The index can be used to look up the result in the history.
        """

        result = self.search(query=query, k=k)
        prev_len = len(self._history)
        splitted = utils.split_search_result_batch(result)
        self._history.extend(splitted)
        return dict(zip(range(prev_len, len(self._history)), splitted))

    def _get_index(self) -> Index:
        return self._index

    def _reset_index(self, index: Index) -> None:
        self._index = index
        self._clear_history()

    index = property(_get_index, _reset_index)

    @property
    def history(self) -> Sequence[SearchResult]:
        "History for looking up the results of previous searches with index handles."

        return self._history

    def _clear_history(self) -> None:
        self._history: list[SearchResult] = []

    @property
    def boundary(self) -> Boundary:
        return self._index.boundary
