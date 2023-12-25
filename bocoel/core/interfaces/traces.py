import abc
from typing import Protocol

from pandas import DataFrame

from .states import State


class Trace(Protocol):
    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __getitem__(self, idx: int, /) -> State:
        ...

    @abc.abstractmethod
    def append(self, state: State, /) -> None:
        ...

    @abc.abstractmethod
    def to_df(self) -> DataFrame:
        ...
