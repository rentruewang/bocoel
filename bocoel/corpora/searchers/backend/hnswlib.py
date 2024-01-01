from typing import Any, Literal

from hnswlib import Index
from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.corpora.interfaces import Distance, Searcher, SearchResult
from bocoel.corpora.searchers import utils

_HnswlibDist = Literal["l2", "ip", "cosine"]


class HnswlibSearcher(Searcher):
    """
    HNSWLIB index. Uses the hnswlib library.

    Score is calculated slightly differently https://github.com/nmslib/hnswlib#supported-distances
    """

    def __init__(
        self, embeddings: NDArray, distance: str | Distance, threads: int = -1
    ) -> None:
        utils.validate_embeddings(embeddings)
        embeddings = utils.normalize(embeddings)

        self._emb = embeddings

        # Would raise ValueError if not a valid distance.
        self._dist = Distance(distance)

        self._bounds = utils.boundaries(embeddings)
        assert self._bounds.shape[1] == 2

        # A public attribute because this can be changed at anytime.
        self.threads = threads

        self._init_index()

    @property
    def distance(self) -> Distance:
        return self._dist

    @property
    def dims(self) -> int:
        return self._emb.shape[1]

    @property
    def bounds(self) -> NDArray:
        return self._bounds

    def _search(self, query: NDArray, k: int = 1) -> SearchResult:
        indices, scores = self._index.knn_query(query, k=k)
        vectors = self._emb[indices]
        return SearchResult(vectors=vectors, scores=scores, indices=indices)

    def _init_index(self) -> None:
        space = self._hnswlib_space(self.distance)
        self._index = Index(space=space, dim=self.dims)
        self._index.init_index(max_elements=len(self._emb))
        self._index.add_items(self._emb, num_threads=self.threads)

    @classmethod
    def from_embeddings(
        cls, embeddings: NDArray, distance: str | Distance, **kwargs: Any
    ) -> Self:
        return cls(embeddings=embeddings, distance=distance, **kwargs)

    @staticmethod
    def _hnswlib_space(distance: Distance) -> _HnswlibDist:
        match distance:
            case Distance.L2:
                return "l2"
            case Distance.INNER_PRODUCT:
                return "ip"
