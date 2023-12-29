from typing import Any

import faiss
from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.corpora.interfaces import Distance, Searcher, SearchResult
from bocoel.corpora.searchers import utils

# TODO: Allow use of GPU for faiss index.


class FaissSearcher(Searcher):
    """
    HNSWLIB index. Uses the hnswlib library.
    """

    def __init__(
        self, embeddings: NDArray, distance: str | Distance, index_string: str
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings to be 2D, got {embeddings.ndim}D.")

        embeddings = utils.normalize(embeddings)

        self._emb = embeddings
        self._dist = distance
        self._dims = embeddings.shape[1]
        self._bounds = utils.boundaries(embeddings)
        assert self._bounds.shape[1] == 2

        self._init_index(index_string=index_string)

    @property
    def distance(self) -> Distance:
        return Distance(self._dist)

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def bounds(self) -> NDArray:
        return self._bounds

    def _search(self, query: NDArray, k: int = 1) -> SearchResult:
        # FIXME: Doing expand_dims because currently only supports 1D non-batched query.
        scores, indices = self._index.search(query[None, :], k)

        scores = scores.squeeze(axis=0)
        indices = indices.squeeze(axis=0)

        vectors = self._emb[indices]
        return SearchResult(vectors=vectors, scores=scores, indices=indices)

    def _init_index(self, index_string: str) -> None:
        metric = self._faiss_metric(self.distance)

        # Using Any as type hint to prevent errors coming up in add / search.
        # Faiss is not type check ready yet.
        # https://github.com/facebookresearch/faiss/issues/2891
        self._index: Any = faiss.index_factory(self.dims, index_string, metric)
        self._index.train(self._emb)
        self._index.add(self._emb)

    @classmethod
    def from_embeddings(
        cls, embeddings: NDArray, distance: str | Distance, **kwargs: Any
    ) -> Self:
        return cls(embeddings=embeddings, distance=distance, **kwargs)

    @staticmethod
    def _faiss_metric(distance: str | Distance) -> Any:
        match distance:
            case Distance.L2:
                return faiss.METRIC_L2
            case Distance.INNER_PRODUCT:
                return faiss.METRIC_INNER_PRODUCT
