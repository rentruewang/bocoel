from __future__ import annotations

import abc
from typing import Mapping, Protocol

from numpy.typing import NDArray
from pandas import DataFrame


class Step(Protocol):
    @abc.abstractproperty
    def candidates(self) -> NDArray:
        ...


class Trace(Protocol):
    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __getitem__(self, idx) -> Step:
        ...

    @abc.abstractmethod
    def append(self, step: Step) -> None:
        ...

    @abc.abstractmethod
    def to_df(self) -> DataFrame:
        ...
