from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray

from bocoel.core.exams.interfaces import Exam
from bocoel.corpora import Distance, FaissIndex, StatefulIndex


class Segregation(Exam):
    def __init__(self, cuda: bool = False, batch_size: int = 64) -> None:
        self._cuda = cuda
        self._batch_size = batch_size

    def _run(self, index: StatefulIndex, results: OrderedDict[int, float]) -> NDArray:
        keys = list(results.keys())
        idx = self._index(index)
        queries = np.array([index.history[i].query for i in keys])

        if queries.ndim != 2:
            raise ValueError(
                f"Expected query to be a 2D vector, got a vector of dim {queries.ndim}."
            )

        outputs = [0]
        for k in range(1, len(keys)):
            queries_upto_k = queries[:k]
            indices = idx.search(queries_upto_k, k=1).indices.squeeze(1)
            unique = np.unique(indices)
            outputs.append(len(unique))

        return np.array(outputs)

    def _index(self, index: StatefulIndex, /) -> FaissIndex:
        # TODO: Only supports L2 for now. Would like more options.
        # Creates a flat index s.t. there is no over head and the solution is exact.
        return FaissIndex(
            embeddings=index.embeddings,
            distance=Distance.L2,
            index_string="Flat",
            cuda=self._cuda,
            batch_size=self._batch_size,
        )
