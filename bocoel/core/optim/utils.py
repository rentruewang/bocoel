from collections.abc import Callable

from numpy.typing import NDArray

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


# TODO: Result is a singleton since k = 1. Support k != 1 in the future.
def evaluate_index(
    *, query: NDArray, index: Index, evaluate_fn: Callable[[SearchResult], float]
) -> State:
    result = index.search(query, k=1)
    evaluation = evaluate_fn(result)
    return State(result=result, score=evaluation)
