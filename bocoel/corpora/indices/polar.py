from typing import Any

from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.corpora.interfaces import Distance, Index, SearchResult


# TODO: Implement polar version of coordinates.
class PolarIndex(Index):
    def _search(self, query: NDArray, k: int = 1) -> SearchResult:
        raise NotImplementedError

    @property
    def embeddings(self) -> NDArray:
        raise NotImplementedError

    @property
    def distance(self) -> Distance:
        raise NotImplementedError

    @property
    def bounds(self) -> NDArray:
        raise NotImplementedError

    @property
    def dims(self) -> int:
        raise NotImplementedError

    @classmethod
    def from_embeddings(
        cls, embeddings: NDArray, distance: str | Distance, **kwargs: Any
    ) -> Self:
        return cls(embeddings=embeddings, distance=distance, **kwargs)
