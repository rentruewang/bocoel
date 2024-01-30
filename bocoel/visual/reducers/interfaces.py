import abc
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame


class Reducer(Protocol):
    size: float
    scores: NDArray
    sample_size: Sequence[int]
    description: Sequence[str]

    @abc.abstractmethod
    def reduce_2d(self, X: NDArray) -> NDArray: ...

    def process(self, X: NDArray) -> DataFrame:
        df = DataFrame()

        df["size"] = self.size
        df["std"] = np.std(self.scores)
        df["sample_size"] = self.sample_size
        df["scores"] = self.scores
        df["Description"] = self.description

        x_reduced = self.reduce_2d(X)

        df["x"] = x_reduced[:, 0]
        df["y"] = x_reduced[:, 1]

        return df
