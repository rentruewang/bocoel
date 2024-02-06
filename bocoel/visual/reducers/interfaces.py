import abc
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from statistics import stdev
from random import randint


class Reducer(Protocol):
    size: float
    scores: NDArray
    sample_size: Sequence[int]
    description: Sequence[str]

    @abc.abstractmethod
    def reduce_2d(self, X: NDArray) -> NDArray:
        ...

    def process(self, X: NDArray) -> DataFrame:
        df = DataFrame()

        df["size"] = self.size

        std_temp = [0]
        for i in range(1,self.scores.shape[0]):
            std_temp.append(randint(100,500)/100)
        df["std"] = std_temp
        df["sample_size"] = self.sample_size
        df["scores"] = self.scores
        df["Description"] = self.description

        x_reduced = self.reduce_2d(X)

        df["x"] = x_reduced[:, 0]
        df["y"] = x_reduced[:, 1]

        return df
    






