from __future__ import annotations

import abc
import typing
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from numpy.typing import NDArray


class Storage(Protocol):
    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @typing.overload
    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Mapping[str, str]:
        ...

    @typing.overload
    @abc.abstractmethod
    def __getitem__(
        self, idx: slice | Sequence[int] | NDArray
    ) -> Sequence[Mapping[str, str]]:
        ...

    @abc.abstractmethod
    def __getitem__(
        self, idx: int | slice | Sequence[int] | NDArray
    ) -> Mapping[str, str] | Sequence[Mapping[str, str]]:
        ...

    @abc.abstractclassmethod
    def from_path(cls, path: str | Path) -> Storage:
        ...
