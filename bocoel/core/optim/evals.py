from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import structlog
from numpy.typing import ArrayLike, NDArray

from bocoel.core.optim.interfaces import Optimizer, QueryEvaluator, SearchEvaluator
from bocoel.corpora import Corpus, SearchResult, SearchResultBatch, StatefulIndex
from bocoel.corpora.indices import utils
from bocoel.models import Adaptor

LOGGER = structlog.get_logger()


def search_eval_func(
    evaluate_fn: Callable[[SearchResultBatch], Sequence[float] | NDArray], /
) -> SearchEvaluator:
    def stateful_eval(sr: Mapping[int, SearchResult], /) -> Mapping[int, float]:
        ordered_dict = OrderedDict(sr)
        batch = utils.join_search_results(ordered_dict.values())
        evaluated = evaluate_fn(batch)
        return dict(zip(ordered_dict.keys(), evaluated))

    return stateful_eval


def query_eval_func(
    index: StatefulIndex, evaluate_fn: SearchEvaluator, k: int = 1
) -> QueryEvaluator:
    LOGGER.debug("Generating evaluation function", index=index, k=k)

    def query_eval(query: ArrayLike, /) -> OrderedDict[int, float]:
        results = index.stateful_search(query, k=k)
        return OrderedDict(evaluate_fn(results))

    return query_eval


def evaluate_index(
    optim_class: type[Optimizer],
    index: StatefulIndex,
    evaluate_fn: SearchEvaluator,
    k: int = 1,
    **kwargs: Any,
) -> Optimizer:
    query_eval = query_eval_func(index, evaluate_fn, k=k)
    return optim_class(query_eval=query_eval, boundary=index.boundary, **kwargs)


def evaluate_corpus(
    optim_class: type[Optimizer],
    corpus: Corpus,
    adaptor: Adaptor,
    k: int = 1,
    **kwargs: Any,
) -> Optimizer:
    def evaluate_fn(sr: SearchResultBatch, /) -> Sequence[float] | NDArray:
        evaluated = adaptor.on_corpus(corpus=corpus, indices=sr.indices)
        assert (
            evaluated.ndim == 2
        ), f"Evaluated should have the dimensions [batch, k]. Got {evaluated.shape}"
        return evaluated.mean(axis=-1)

    return evaluate_index(
        optim_class=optim_class,
        index=corpus.index,
        evaluate_fn=search_eval_func(evaluate_fn),
        k=k,
        **kwargs,
    )
