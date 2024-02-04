import functools
import warnings
from typing import Any

from numpy.typing import NDArray

from bocoel.corpora.indices import utils
from bocoel.corpora.indices.interfaces import (
    Boundary,
    Distance,
    Index,
    IndexedArray,
    InternalResult,
)


@functools.cache
def _faiss():
    # Optional dependency. Faiss also spits out deprecation warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        import faiss

    return faiss


class FaissIndex(Index):
    """
    Faiss index. Uses the faiss library.
    """

    def __init__(
        self,
        embeddings: NDArray,
        distance: str | Distance,
        index_string: str,
        cuda: bool = False,
        batch_size: int = 64,
    ) -> None:
        utils.validate_embeddings(embeddings)
        embeddings = utils.normalize(embeddings)
        self._emb = embeddings

        self._batch_size = batch_size
        self._dist = Distance.lookup(distance)
        self._boundary = utils.boundaries(embeddings)

        self._index_string = index_string
        self._init_index(index_string=index_string, cuda=cuda)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._index_string}, {self.dims})"

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
        distances, indices = self._index.search(query, k)
        return InternalResult(distances=distances, indices=indices)

    def _init_index(self, index_string: str, cuda: bool) -> None:
        metric = self._faiss_metric(self.distance)

        # Using Any as type hint to prevent errors coming up in add / search.
        # Faiss is not type check ready yet.
        # https://github.com/facebookresearch/faiss/issues/2891

        index: Any = _faiss().index_factory(self.dims, index_string, metric)
        index.train(self._emb)
        index.add(self._emb)

        if cuda:
            index = _faiss().index_cpu_to_all_gpus(index)

        self._index = index

    @staticmethod
    def _faiss_metric(distance: Distance) -> Any:
        match distance:
            case Distance.L2:
                return _faiss().METRIC_L2
            case Distance.INNER_PRODUCT:
                return _faiss().METRIC_INNER_PRODUCT
