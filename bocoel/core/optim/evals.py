import abc
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import structlog
from numpy.typing import ArrayLike, NDArray

from bocoel.corpora import SearchResult, SearchResultBatch, StatefulIndex
from bocoel.corpora.indices import utils

LOGGER = structlog.get_logger()


class ResultEvaluator(Protocol):
    @abc.abstractmethod
    def __call__(self, sr: Mapping[int, SearchResult], /) -> Mapping[int, float]:
        ...


class QueryEvaluator(Protocol):
    @abc.abstractmethod
    def __call__(self, query: ArrayLike, /) -> OrderedDict[int, float]:
        """
        Evaluates the given batched query.
        The order of the results must be kept in the original order.
        """

        ...


def stateful_eval_func(
    evaluate_fn: Callable[[SearchResultBatch], Sequence[float] | NDArray], /
) -> ResultEvaluator:
    def stateful_eval(sr: Mapping[int, SearchResult], /) -> Mapping[int, float]:
        ordered_dict = OrderedDict(sr)
        batch = utils.join_search_results(ordered_dict.values())
        evaluated = evaluate_fn(batch)
        return dict(zip(ordered_dict.keys(), evaluated))

    return stateful_eval


def query_eval_func(
    index: StatefulIndex, evaluate_fn: ResultEvaluator, k: int = 1
) -> QueryEvaluator:
    LOGGER.debug("Generating evaluation function", index=index, k=k)

    def query_eval(query: ArrayLike, /) -> OrderedDict[int, float]:
        results = index.stateful_search(query, k=k)
        return OrderedDict(evaluate_fn(results))

    return query_eval
