from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from .interfaces import Processor


# TODO: Attach real data
class PCAPreprocessor(Processor):
    def __init__(
        self,
        scores: NDArray = np.random.rand(100),
        size: float = 0.5,
        # FIXME: Should be a sequence of floats.
        sample_size: Sequence = tuple(range(1, 101)),
        desc: Sequence = (),
        algo: str = "PCA",
    ):
        self.scores = scores
        self.size = size
        self.sample_size = sample_size
        self._algo = algo
        self.description = (
            desc if desc else ["Fake prompt number {}".format(i) for i in range(1, 101)]
        )

    def reduce_2d(self, X: NDArray) -> NDArray:
        func = PCA(n_components=2, svd_solver="full").fit_transform
        return func(X)
