from typing import NamedTuple

from numpy.typing import NDArray


class State(NamedTuple):
    candidates: NDArray
    scores: float | NDArray
