from __future__ import annotations

import abc
from typing import Protocol

from numpy.typing import NDArray


class Embedder(Protocol):
    dims: int

    @abc.abstractmethod
    def encode(self, text: str) -> NDArray:
        ...
