from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray

from bocoel.core.exams.interfaces import Exam
from bocoel.corpora import StatefulIndex


class Segregation(Exam):
    def _run(self, index: StatefulIndex, results: OrderedDict[int, float]) -> NDArray:
        keys = list(results.keys())
        idx = index.index
        queries = np.array([index[i].query for i in keys])

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
