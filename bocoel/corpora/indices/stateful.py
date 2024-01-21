from collections.abc import Mapping, Sequence

from numpy.typing import ArrayLike, NDArray

from bocoel.corpora.indices import utils
from bocoel.corpora.indices.interfaces import (
    Boundary,
    Distance,
    Index,
    IndexedArray,
    SearchResult,
)
from bocoel.corpora.indices.interfaces.results import InternalResult


class StatefulIndex(Index):
    def __init__(self, index: Index) -> None:
        self._index = index
        self._clear_history()

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
    def _embeddings(self) -> NDArray | IndexedArray:
        return self._index._embeddings

    @property
    def distance(self) -> Distance:
        return self._index.distance

    def stateful_search(
        self, query: ArrayLike, k: int = 1
    ) -> Mapping[int, SearchResult]:
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
        return self._history

    def _clear_history(self) -> None:
        self._history: list[SearchResult] = []

    @property
    def boundary(self) -> Boundary:
        return self._index.boundary
