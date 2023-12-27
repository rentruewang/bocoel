from pandas import DataFrame

from bocoel.core.interfaces import State, Trace


class ListTrace(Trace):
    def __init__(self) -> None:
        self.__list: list[State] = []

    def __len__(self) -> int:
        return len(self.__list)

    def __getitem__(self, idx: int, /) -> State:
        return self.__list[idx]

    def append(self, state: State, /) -> None:
        self.__list.append(state)

    def to_df(self) -> DataFrame:
        return DataFrame.from_records([self[i].__dict__ for i in range(len(self))])
