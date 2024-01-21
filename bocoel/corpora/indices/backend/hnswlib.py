from typing import Literal

from numpy.typing import NDArray

from bocoel.corpora.indices import utils
from bocoel.corpora.indices.interfaces import (
    Boundary,
    Distance,
    Index,
    IndexedArray,
    InternalResult,
)

_HnswlibDist = Literal["l2", "ip", "cosine"]


class HnswlibIndex(Index):
    """
    HNSWLIB index. Uses the hnswlib library.

    Score is calculated slightly differently https://github.com/nmslib/hnswlib#supported-distances
    """

    def __init__(
        self,
        embeddings: NDArray,
        distance: str | Distance,
        threads: int = -1,
        batch_size: int = 64,
    ) -> None:
        utils.validate_embeddings(embeddings)
        embeddings = utils.normalize(embeddings)

        self._emb = embeddings

        # Would raise ValueError if not a valid distance.
        self._dist = Distance.lookup(distance)
        self._batch_size = batch_size

        self._boundary = utils.boundaries(embeddings)

        # A public attribute because this can be changed at anytime.
        self.threads = threads

        self._init_index()

    @property
    def batch(self) -> int:
        return self._batch_size

    @property
    def _embeddings(self) -> NDArray | IndexedArray:
        return self._emb

    @property
    def distance(self) -> Distance:
        return self._dist

    @property
    def dims(self) -> int:
        return self._emb.shape[1]

    @property
    def boundary(self) -> Boundary:
        return self._boundary

    def _search(self, query: NDArray, k: int = 1) -> InternalResult:
        indices, distances = self._index.knn_query(query, k=k, num_threads=self.threads)
        return InternalResult(indices=indices, distances=distances)

    def _init_index(self) -> None:
        # Optional dependency.
        from hnswlib import Index as _HnswlibIndex

        space = self._hnswlib_space(self.distance)
        self._index = _HnswlibIndex(space=space, dim=self.dims)
        self._index.init_index(max_elements=len(self._emb))
        self._index.add_items(self._emb, num_threads=self.threads)

    @staticmethod
    def _hnswlib_space(distance: Distance) -> _HnswlibDist:
        match distance:
            case Distance.L2:
                return "l2"
            case Distance.INNER_PRODUCT:
                return "ip"
