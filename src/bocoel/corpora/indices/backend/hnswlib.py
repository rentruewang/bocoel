# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from typing import Literal

from numpy.typing import NDArray

from bocoel.corpora.indices import utils
from bocoel.corpora.indices.interfaces import Distance, Index, InternalResult

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
        *,
        normalize: bool = True,
        threads: int = -1,
        batch_size: int = 64,
    ) -> None:
        """
        Initializes the HNSWLIB index.

        Parameters:
            embeddings: The embeddings to index.
            distance: The distance metric to use.
            normalize: Whether to normalize the embeddings.
            threads: The number of threads to use.
            batch_size: The batch size to use for searching.

        Raises:
            ValueError: If the distance is not supported.
        """

        if normalize:
            embeddings = utils.normalize(embeddings)

        self.__embeddings = embeddings

        # Would raise ValueError if not a valid distance.
        self._dist = Distance.lookup(distance)
        self._batch_size = batch_size

        # A public attribute because this can be changed at anytime.
        self.threads = threads

        self._init_index()

    @property
    def batch(self) -> int:
        return self._batch_size

    @property
    def data(self) -> NDArray:
        return self.__embeddings

    @property
    def distance(self) -> Distance:
        return self._dist

    def _search(self, query: NDArray, k: int = 1) -> InternalResult:
        indices, distances = self._index.knn_query(query, k=k, num_threads=self.threads)
        return InternalResult(indices=indices, distances=distances)

    def _init_index(self) -> None:
        # Optional dependency.
        from hnswlib import Index as _HnswlibIndex

        space = self._hnswlib_space(self.distance)
        self._index = _HnswlibIndex(space=space, dim=self.dims)
        self._index.init_index(max_elements=len(self.data))
        self._index.add_items(self.data, num_threads=self.threads)

    @staticmethod
    def _hnswlib_space(distance: Distance) -> _HnswlibDist:
        match distance:
            case Distance.L2:
                return "l2"
            case Distance.INNER_PRODUCT:
                return "ip"
