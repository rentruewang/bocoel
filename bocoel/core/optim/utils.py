from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self

from bocoel.core.optim.interfaces import State
from bocoel.corpora import Index, SearchResult


class RemainingSteps:
    def __init__(self, count: int) -> None:
        self._count = count

    @property
    def count(self) -> int:
        return self._count

    def step(self) -> None:
        self._count -= 1

    @property
    def done(self) -> bool:
        # This would never be true if renaming steps if < 0 at first.
        return self._count == 0

    @classmethod
    def infinite(cls) -> Self:
        return cls(count=-1)


def evaluate_index(
    *,
    query: ArrayLike,
    index: Index,
    evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
    k: int = 1,
) -> State:
    query = np.array(query)
    result = index.search(query, k=k)
    evaluation = np.array(evaluate_fn(result))

    return State(result=result, evaluation=evaluation)
