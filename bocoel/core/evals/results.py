import abc
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import structlog
from numpy.typing import ArrayLike, NDArray

from bocoel.corpora import Index, SearchResult

from .queries import QueryEvaluator

LOGGER = structlog.get_logger()


class ResultEvaluator(Protocol):
    @abc.abstractmethod
    def __call__(self, sr: SearchResult, /) -> Sequence[float] | NDArray:
        ...


def index_eval_func(
    index: Index, evaluate_fn: ResultEvaluator, k: int = 1
) -> QueryEvaluator:
    LOGGER.debug("Generating evaluation function", index=index, k=k)

    def query_evaluator(query: ArrayLike, /) -> Sequence[float] | NDArray:
        sr = index.search(np.array(query), k=k)
        evaluation = evaluate_fn(sr)
        return evaluation

    return query_evaluator
