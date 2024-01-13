from collections.abc import Callable, Sequence

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
) -> Sequence[State]:
    """
    Evaluates indices on a batch of queries.

    Parameters
    ----------

    `query : ArrayLike`
    The query to evaluate on. Should be a batch of queries.

    `index : Index`
    The index to evaluate on.

    `evaluate_fn : Callable[[SearchResult], Sequence[float] | NDArray]`
    The function to evaluate the index with.
    Takes in a search result and returns a sequence of scores.

    `k : int`
    Nearest neighbors to retrieve.

    Returns
    -------

    A sequence of states.
    """

    sr = index.search(query, k=k)
    evaluation = evaluate_fn(sr)

    return [
        State(query=q, vectors=v, distances=d, indices=i, evaluation=e)
        for q, v, d, i, e in zip(
            sr.query, sr.vectors, sr.distances, sr.indices, evaluation
        )
    ]
