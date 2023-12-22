from __future__ import annotations

from typing import Mapping, Protocol

from numpy.typing import NDArray


class State(Protocol):
    embedding: NDArray
    retrieved: NDArray
    index: int
    scores: Mapping[str, float]
