from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from bocoel.core.interfaces import State
from bocoel.corpora import Corpus, Searcher
from bocoel.models import Evaluator
from bocoel.models import utils as model_utils


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


def evaluate_query(*, query: NDArray, corpus: Corpus, evaluator: Evaluator) -> State:
    # FIXME: Result is a singleton since k = 1. Support batch in the future.
    result = corpus.searcher.search(query)
    indices: int = result.indices.item()
    vectors = result.vectors

    # FIXME: This is a temporary hack to only evaluate one query.
    evaluation = model_utils.evaluate_on_corpus(
        evaluator=evaluator, corpus=corpus, indices=[indices]
    )[0]
    return State(candidates=query.squeeze(), actual=vectors, scores=evaluation)


def evaluate_searcher(
    *,
    query: NDArray,
    searcher: Searcher,
    evaluate_fn: Callable[[Mapping[str, str]], float]
) -> State:
    # FIXME: Result is a singleton since k = 1. Support batch in the future.
    result = searcher.search(query)
    indices: int = result.indices.item()

    raise NotImplementedError
