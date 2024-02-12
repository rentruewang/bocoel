from collections import OrderedDict

import alive_progress as ap
import numpy as np
import structlog
from numpy.typing import NDArray

from bocoel.core.exams.interfaces import Exam
from bocoel.corpora import StatefulIndex

LOGGER = structlog.get_logger()


class Segregation(Exam):
    """
    This test measures how evenly distributed the texts in the corpus are.
    The Segregation exam that computes the segregation of the corpus.
    A good corpus shall score high on this exam.
    """

    def _run(self, index: StatefulIndex, results: OrderedDict[int, float]) -> NDArray:
        LOGGER.info("Running Segregation exam", num_results=len(results))

        keys = list(results.keys())
        idx = index.index
        queries = np.array([index[i].query for i in keys])

        if queries.ndim != 2:
            raise ValueError(
                f"Expected query to be a 2D vector, got a vector of dim {queries.ndim}."
            )

        outputs = [0]
        for k in ap.alive_it(range(1, len(keys)), title="Running Segregation exam"):
            queries_upto_k = queries[:k]
            indices = idx.search(queries_upto_k, k=1).indices.squeeze(1)
            unique = np.unique(indices)
            outputs.append(len(unique))

        return np.array(outputs)
