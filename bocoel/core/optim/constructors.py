from collections.abc import Sequence
from typing import Any

from numpy.typing import NDArray

from bocoel.core.optim import evals
from bocoel.core.optim.interfaces import Optimizer, SearchEvaluator
from bocoel.corpora import Corpus, SearchResultBatch, StatefulIndex
from bocoel.models import Adaptor, LanguageModel


def evaluate_index(
    optim_class: type[Optimizer],
    index: StatefulIndex,
    evaluate_fn: SearchEvaluator,
    **kwargs: Any,
) -> Optimizer:
    query_eval = evals.query_eval_func(index, evaluate_fn)
    return optim_class(query_eval, **kwargs)


def evaluate_corpus(
    optim_class: type[Optimizer],
    corpus: Corpus,
    lm: LanguageModel,
    adaptor: Adaptor,
    **kwargs: Any,
) -> Optimizer:
    def evaluate_fn(sr: SearchResultBatch, /) -> Sequence[float] | NDArray:
        evaluated = adaptor.on_corpus(corpus=corpus, lm=lm, indices=sr.indices)
        assert (
            evaluated.ndim == 2
        ), f"Evaluated should have the dimensions [batch, k]. Got {evaluated.shape}"
        return evaluated.mean(axis=-1)

    return evaluate_index(
        optim_class=optim_class,
        index=corpus.index,
        evaluate_fn=evals.search_eval_func(evaluate_fn),
        **kwargs,
    )
