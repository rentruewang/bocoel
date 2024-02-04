import abc
import typing
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Protocol

import typeguard

from bocoel import common


class Storage(Protocol):
    """
    Storage is responsible for storing the data.
    This can be thought of as a table.
    """

    def __repr__(self) -> str:
        name = common.remove_base_suffix(self, Storage)
        return f"{name}({list(self.keys())}, {len(self)})"

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of rows in the storage.
        """

        ...

    @typing.overload
    def __getitem__(self, idx: int) -> Mapping[str, Any]: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> Mapping[str, Sequence[Any]]: ...

    def __getitem__(
        self, idx: int | slice | Sequence[int]
    ) -> Mapping[str, Any] | Mapping[str, Sequence[Any]]:
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

    @abc.abstractmethod
    def keys(self) -> Collection[str]: ...

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
