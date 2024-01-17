from collections.abc import Sequence

import numpy as np
from numpy import random
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA

from .interfaces import Reducer


class PCAReducer(Reducer):
    def __init__(
        self,
        scores: ArrayLike = random.rand(100),
        size: float = 0.5,
        sample_size: ArrayLike = np.arange(1, 101).tolist(),
        desc: Sequence[str] = (),
        algo: str = "PCA",
    ):
        self.scores = np.array(scores)
        self.size = size
        self.sample_size = np.array(sample_size).tolist()
        self._algo = algo
        self.description = (
            desc if desc else ["Fake prompt number {}".format(i) for i in range(1, 101)]
        )

    def reduce_2d(self, X: NDArray) -> NDArray:
        func = PCA(n_components=2, svd_solver="full").fit_transform
        return func(X)
