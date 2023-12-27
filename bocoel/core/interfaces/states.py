from typing import NamedTuple

from numpy.typing import NDArray


class State(NamedTuple):
    candidates: NDArray
    actual: NDArray
    scores: float | NDArray
