from typing import Any

from bocoel import (
    Adaptor,
    AxServiceOptimizer,
    Corpus,
    KMeansOptimizer,
    KMedoidsOptimizer,
    Optimizer,
    core,
)
from bocoel.common import StrEnum

from . import common


class OptimizerName(StrEnum):
    AX_SERVICE = "AX_SERVICE"
    KMEANS = "KMEANS"
    KMEDOIDS = "KMEDOIDS"


def optimizer_factory(
    name: str | OptimizerName, /, *, corpus: Corpus, adaptor: Adaptor, **kwargs: Any
) -> Optimizer:
    name = OptimizerName.lookup(name)

    match name:
        case OptimizerName.AX_SERVICE:
            return common.correct_kwargs(core.evaluate_corpus)(
                AxServiceOptimizer, corpus=corpus, adaptor=adaptor, **kwargs
            )
        case OptimizerName.KMEANS:
            return common.correct_kwargs(core.evaluate_corpus)(
                KMeansOptimizer, corpus=corpus, adaptor=adaptor, **kwargs
            )
        case OptimizerName.KMEDOIDS:
            return common.correct_kwargs(core.evaluate_corpus)(
                KMedoidsOptimizer, corpus=corpus, adaptor=adaptor, **kwargs
            )
        case _:
            raise ValueError(f"Unknown optimizer name: {name}")
