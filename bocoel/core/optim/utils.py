import numpy as np
from numpy.typing import NDArray

from bocoel.core.interfaces import State
from bocoel.corpora import Corpus
from bocoel.models import Evaluator, LanguageModel


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


def check_bounds(corpus: Corpus) -> None:
    bounds = corpus.searcher.bounds

    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("The bound is not valid")

    lower, upper = bounds.T
    if np.any(lower > upper):
        raise ValueError("lower > upper at some points")


def evaluate_query(
    *, query: NDArray, corpus: Corpus, lm: LanguageModel, evaluator: Evaluator
) -> State:
    # Result is a singleton since k = 1.
    result = corpus.searcher.search(query)
    indices: int = result.indices.item()
    vectors = result.vectors

    # FIXME: This is a temporary hack to only evaluate one query.
    evaluation = evaluator.evaluate(lm, corpus, indices=[indices])[0]
    return State(candidates=query.squeeze(), actual=vectors, scores=evaluation)
