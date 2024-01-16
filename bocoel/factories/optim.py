from typing import Any

from bocoel import (
    AxServiceOptimizer,
    Corpus,
    Evaluator,
    LanguageModel,
    Optimizer,
    ScikitLearnOptimizer,
)
from bocoel.common import StrEnum

from . import common


class OptimizerName(StrEnum):
    AX_SERVICE = "AX_SERVICE"
    KMEANS = "KMEANS"


def optimizer_factory(
    name: str | OptimizerName,
    /,
    *,
    corpus: Corpus,
    lm: LanguageModel,
    evaluator: Evaluator,
    **kwargs: Any,
) -> Optimizer:
    name = OptimizerName.lookup(name)

    match name:
        case OptimizerName.AX_SERVICE:
            return common.correct_kwargs(AxServiceOptimizer.evaluate_corpus)(
                corpus=corpus, lm=lm, evaluator=evaluator, **kwargs
            )
        case OptimizerName.KMEANS:
            return common.correct_kwargs(ScikitLearnOptimizer.evaluate_corpus)(
                corpus=corpus, lm=lm, evaluator=evaluator, **kwargs
            )
        case _:
            raise ValueError(f"Unknown optimizer name: {name}")
