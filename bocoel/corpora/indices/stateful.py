from collections.abc import Mapping, Sequence

from numpy.typing import ArrayLike

from bocoel.corpora.indices import utils
from bocoel.corpora.indices.interfaces import Index, SearchResult


class StatefulIndex:
    def __init__(self, index: Index) -> None:
        self._index = index
        self._clear_history()

    def __len__(self) -> int:
        return len(self._history)

    def __getitem__(self, key: int, /) -> SearchResult:
        return self._history[key]

    def stateful_search(
        self, query: ArrayLike, k: int = 1
    ) -> Mapping[int, SearchResult]:
        result = self._index.search(query=query, k=k)
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
