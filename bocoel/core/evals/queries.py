import abc
from collections.abc import Sequence
from typing import Protocol

from numpy.typing import ArrayLike, NDArray


class QueryEvaluator(Protocol):
    @abc.abstractmethod
    def __call__(self, query: ArrayLike, /) -> Sequence[float] | NDArray:
        ...
