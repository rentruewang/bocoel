from __future__ import annotations

import abc
from pathlib import Path
from typing import AbstractSet, Mapping, Protocol


class Storage(Protocol):
    """
    Storage is responsible for storing the data.
    This can be thought of as a table.
    """

    @abc.abstractmethod
    def keys(self) -> AbstractSet:
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
