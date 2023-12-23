from __future__ import annotations

import abc
from typing import Protocol, Tuple

from numpy.typing import NDArray


class Bounded(Protocol):
    @abc.abstractmethod
    def bounds(self) -> NDArray:
        ...

    def is_valid(self) -> bool:
        bounds = self.bounds()
        return bounds.ndim == 2 and bounds.shape[1] == 2

    def __len__(self) -> int:
        return len(self.bounds())

    def __getitem__(self, idx: int) -> Tuple[float, float]:
        lower, upper = self.bounds()[idx]
        return lower, upper

    @property
    def ndim(self) -> int:
        self.__must_be_valid()
        return self.bounds().shape[0]

    @property
    def lower(self) -> NDArray:
        self.__must_be_valid()
        return self.bounds()[:, 0]

    @property
    def upper(self) -> NDArray:
        self.__must_be_valid()
        return self.bounds()[:, 1]

    def __must_be_valid(self) -> None:
        if not self.is_valid():
            raise ValueError("The bound is not valid.")


class Optimizer(Protocol):
    @abc.abstractmethod
    def optimize(self, bound: Bounded) -> None:
        ...
