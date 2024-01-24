import abc
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Protocol


class Storage(Protocol):
    """
    Storage is responsible for storing the data.
    This can be thought of as a table.
    """

    @abc.abstractmethod
    def keys(self) -> Collection[str]:
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of rows in the storage.
        """

        ...

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        """
        Returns the row at the given index.
        """

        ...

    @abc.abstractmethod
    def get(self, key: str) -> Sequence[Any]:
        """
        Get the entire column by given key.
        """

        ...
