from __future__ import annotations

import abc
from typing import Container, Mapping, Protocol, Sequence


class Storage(Protocol):
    """
    Storage is responsible for storing the data.
    This can be thought of as a table.
    """

    @abc.abstractmethod
    def keys(self) -> Container:
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of rows in the storage.
        """

        ...

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Mapping[str, str]:
        """
        Returns the row at the given index.
        """
        ...

    def get(self, key: str) -> Sequence[str]:
        return [self[i][key] for i in range(len(self))]
