from __future__ import annotations

import abc
from pathlib import Path
from typing import Mapping, Protocol


class Storage(Protocol):
    """
    Storage is responsible for storing the data.
    This can be thought of as a table.
    """

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

    @abc.abstractmethod
    @classmethod
    def from_path(cls, path: str | Path) -> Storage:
        """
        Construct a storage from a local file.
        """
        ...
