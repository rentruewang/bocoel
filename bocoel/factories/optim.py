from typing import Any

from bocoel import (
    Adaptor,
    AxServiceOptimizer,
    Corpus,
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
    adaptor: Adaptor,
    **kwargs: Any,
) -> Optimizer:
    name = OptimizerName.lookup(name)

    match name:
        case OptimizerName.AX_SERVICE:
            return common.correct_kwargs(AxServiceOptimizer.evaluate_corpus)(
                corpus=corpus, lm=lm, adaptor=adaptor, **kwargs
            )
        case OptimizerName.KMEANS:
            return common.correct_kwargs(ScikitLearnOptimizer.evaluate_corpus)(
                corpus=corpus, lm=lm, adaptor=adaptor, **kwargs
            )
        case _:
            raise ValueError(f"Unknown optimizer name: {name}")
