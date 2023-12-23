from __future__ import annotations

import abc
from typing import Protocol

from numpy.typing import NDArray


class Bounded(Protocol):
    bounds: NDArray

    def is_valid(self) -> bool:
        return len(self.bounds) == 2 and self.bounds.ndim == 2

    @property
    def ndim(self) -> int:
        self.__must_be_valid()
        return self.bounds.shape[1]

    @property
    def lower(self) -> NDArray:
        self.__must_be_valid()
        return self.bounds[0]

    @property
    def upper(self) -> NDArray:
        self.__must_be_valid()
        return self.bounds[1]

    def __must_be_valid(self) -> None:
        if not self.is_valid():
            raise ValueError("The bound is not valid.")


class Optimizer(Protocol):
    @abc.abstractmethod
    def optimize(self, bound: Bounded) -> None:
        ...
