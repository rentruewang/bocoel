from typing import Any

from bocoel import (
    AxServiceOptimizer,
    Corpus,
    Evaluator,
    KMeansOptimizer,
    LanguageModel,
    Optimizer,
)
from bocoel.common import StrEnum


class OptimizerName(StrEnum):
    AX_SERVICE = "AX_SERVICE"
    KMEANS = "KMEANS"


def optimizer_factory(
    name: str | OptimizerName,
    /,
    corpus: Corpus,
    lm: LanguageModel,
    evaluator: Evaluator,
    **kwargs: Any,
) -> Optimizer:
    name = OptimizerName.lookup(name)

    match name:
        case OptimizerName.AX_SERVICE:
            return AxServiceOptimizer.evaluate_corpus(
                corpus=corpus, lm=lm, evaluator=evaluator, **kwargs
            )
        case OptimizerName.KMEANS:
            return KMeansOptimizer.evaluate_corpus(
                corpus=corpus, lm=lm, evaluator=evaluator, **kwargs
            )
        case _:
            raise ValueError(f"Unknown optimizer name: {name}")
