from __future__ import annotations

from typing import List, NamedTuple

from numpy.typing import NDArray
from pandas import DataFrame

from bocoel.core.interfaces import Step, Trace


class ImmutableStep(NamedTuple):
    candidates: NDArray


class ListTrace(Trace):
    def __init__(self) -> None:
        self.__list: List[Step] = []

    def __len__(self) -> int:
        return len(self.__list)

    def __getitem__(self, idx: int) -> Step:
        return self.__list[idx]

    def append(self, step: Step) -> None:
        self.__list.append(step)

    def to_df(self) -> DataFrame:
        return DataFrame.from_records([self[i].__dict__ for i in range(len(self))])
