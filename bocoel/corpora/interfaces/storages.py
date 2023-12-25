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

    @abc.abstractmethod
    def get(self, key: str) -> Sequence[str]:
        """
        Get the entire column by given key.
        """

        ...
