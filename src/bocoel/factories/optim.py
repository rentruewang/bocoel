# Copyright (c) BoCoEL Authors - All Rights Reserved

from typing import Any

from bocoel import (
    Adaptor,
    AxServiceOptimizer,
    BruteForceOptimizer,
    Corpus,
    CorpusEvaluator,
    KMeansOptimizer,
    KMedoidsOptimizer,
    Optimizer,
    RandomOptimizer,
    UniformOptimizer,
)

from . import common

__all__ = ["optimizer"]


@common.correct_kwargs
def optimizer(
    name: str, /, *, corpus: Corpus, adaptor: Adaptor, **kwargs: Any
) -> Optimizer:
    """
    Create an optimizer instance.

    Parameters:
        name: The name of the optimizer.
        corpus: The corpus to optimize.
        adaptor: The adaptor to use.
        **kwargs: Additional keyword arguments to pass to the optimizer.
            See the documentation for the specific optimizer for details.

    Returns:
        The optimizer instance.

    Raises:
        ValueError: If the name is unknown.
    """

    klass: type[Optimizer]

    match name:
        case "BAYESIAN":
            klass = AxServiceOptimizer
        case "KMEANS":
            klass = KMeansOptimizer
        case "KMEDOIDS":
            klass = KMedoidsOptimizer
        case "BRUTE":
            klass = BruteForceOptimizer
        case "RANDOM":
            klass = RandomOptimizer
        case "UNIFORM":
            klass = UniformOptimizer
        case _:
            raise ValueError(f"Unknown optimizer name: {name}")

    corpus_evaluator = CorpusEvaluator(corpus=corpus, adaptor=adaptor)
    return klass(index_eval=corpus_evaluator, index=corpus.index, **kwargs)
