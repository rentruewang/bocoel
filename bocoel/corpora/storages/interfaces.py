import abc
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, Protocol

import typeguard
from numpy.typing import NDArray


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

    def __getitem__(
        self, idx: int | slice | Sequence[int]
    ) -> Mapping[str, Sequence[Any]]:
        if isinstance(idx, int):
            return self._getitem(idx)
        elif isinstance(idx, slice):
            slice_range = range(*idx.indices(len(self)))
            return self.collate([self._getitem(i) for i in slice_range])
        elif isinstance(idx, Sequence):
            typeguard.check_type("idx", idx, Sequence[int])
            return self.collate([self._getitem(i) for i in idx])
        else:
            raise TypeError(f"Index must be int or sequence, got {type(idx)}")

    @abc.abstractmethod
    def _getitem(self, idx: int) -> Mapping[str, Any]:
        """
        Returns the row at the given index.
        """

        ...

    def map(
        self,
        idx: int | Sequence[int],
        transform: Callable[
            [Mapping[str, Sequence[Any]]], Mapping[str, Sequence[Any]]
        ] = lambda x: x,
    ) -> Mapping[str, Sequence[Any]]:
        """
        Convert the entire column by given key.

        Parameters
        ----------

        `*key: str`
        Column names to retrieve.

        `concat: Callable[..., Any] | None = None`
        Function to concatenate the columns.

        Returns
        -------

        A sequence of values, corresponding to different keys.
        If `concat` is specified, then the values are concatenated using the function.
        """

        data = self[idx]
        return transform(data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({len(self)}) {self.keys()}"

    @staticmethod
    def collate(mappings: Sequence[Mapping[str, Any]]) -> Mapping[str, Sequence[Any]]:
        if len(mappings) == 0:
            return {}

        first = mappings[0]
        keys = first.keys()

        result = {}

        for key in keys:
            extracted = [item[key] for item in mappings]
            result[key] = extracted

        return result
