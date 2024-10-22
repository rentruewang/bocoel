import importlib
import warnings
from typing import Any, Optional, Union
from numpy.typing import NDArray
from bocoel.corpora.indices import utils
from bocoel.corpora.indices.interfaces import Distance, Index, InternalResult

def _faiss():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return importlib.import_module("faiss")

class FaissIndex(Index):
    """
    Faiss index. Uses the faiss library for fast similarity search.

    Attributes:
        embeddings (NDArray): Input data for the index.
        distance (Distance): Distance metric used in searches.
        index_string (str): Faiss index type (e.g., 'Flat', 'IVF').
        cuda (bool): Whether to use GPU acceleration.
        batch_size (int): Batch size for searching.
    """

    def __init__(
        self,
        embeddings: NDArray,
        distance: Union[str, Distance],
        *,
        normalize: bool = True,
        index_string: str,
        cuda: bool = False,
        batch_size: int = 64,
    ) -> None:
        if normalize:
            embeddings = utils.normalize(embeddings)

        self.__embeddings = embeddings
        self._batch_size = batch_size
        self._dist = Distance.lookup(distance)
        self._index_string = index_string

        self._init_index(index_string=index_string, cuda=cuda)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._index_string}, {self.dims})"

    @property
    def batch(self) -> int:
        return self._batch_size

    @batch.setter
    def batch(self, size: int) -> None:
        if size <= 0:
            raise ValueError("Batch size must be positive.")
        self._batch_size = size

    @property
    def data(self) -> NDArray:
        return self.__embeddings

    @property
    def distance(self) -> Distance:
        return self._dist

    @property
    def dims(self) -> int:
        return self.__embeddings.shape[1]

    def _search(self, query: NDArray, k: int = 1) -> InternalResult:
        if query.shape[0] > self._batch_size:
            results = [
                self._index.search(query[i:i + self._batch_size], k)
                for i in range(0, query.shape[0], self._batch_size)
            ]
            distances, indices = map(np.concatenate, zip(*results))
        else:
            distances, indices = self._index.search(query, k)
        return InternalResult(distances=distances, indices=indices)

    def _init_index(self, index_string: str, cuda: bool) -> None:
        try:
            metric = self._faiss_metric(self.distance)
            index: Any = _faiss().index_factory(self.dims, index_string, metric)
            index.train(self.data)
            index.add(self.data)

            if cuda:
                index = _faiss().index_cpu_to_all_gpus(index)

            self._index = index
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Faiss index: {e}")

    @staticmethod
    def _faiss_metric(distance: Distance) -> int:
        metrics = {
            Distance.L2: _faiss().METRIC_L2,
            Distance.INNER_PRODUCT: _faiss().METRIC_INNER_PRODUCT,
        }
        return metrics[distance]

    def close(self) -> None:
        """Releases GPU resources if CUDA is used."""
        if hasattr(self._index, 'reset'):
            self._index.reset()
