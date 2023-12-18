import abc
from typing import Protocol

from .states import State


class History(Protocol):
    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> State:
        ...

    @abc.abstractmethod
    def append(self, entry: State) -> None:
        ...
