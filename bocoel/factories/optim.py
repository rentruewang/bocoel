from typing import Any

from bocoel import (
    Adaptor,
    AxServiceOptimizer,
    BruteForceOptimizer,
    Corpus,
    KMeansOptimizer,
    KMedoidsOptimizer,
    Optimizer,
    RandomOptimizer,
    UniformOptimizer,
    core,
)
from bocoel.common import StrEnum

from . import common


class OptimizerName(StrEnum):
    """
    The names of the optimizers.
    """

    BAYESIAN = "BAYESIAN"
    "Corresponds to `AxServiceOptimizer`."

    KMEANS = "KMEANS"
    "Corresponds to `KMeansOptimizer`."

    KMEDOIDS = "KMEDOIDS"
    "Corresponds to `KMedoidsOptimizer`."

    RANDOM = "RANDOM"
    "Corresponds to `RandomOptimizer`."

    BRUTE = "BRUTE"
    "Corresponds to `BruteForceOptimizer`."

    UNIFORM = "UNIFORM"
    "Corresponds to `UniformOptimizer`."


def optimizer(
    name: str | OptimizerName, /, *, corpus: Corpus, adaptor: Adaptor, **kwargs: Any
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

    name = OptimizerName.lookup(name)

    klass: type[Optimizer]

    match name:
        case OptimizerName.BAYESIAN:
            klass = AxServiceOptimizer
        case OptimizerName.KMEANS:
            klass = KMeansOptimizer
        case OptimizerName.KMEDOIDS:
            klass = KMedoidsOptimizer
        case OptimizerName.BRUTE:
            klass = BruteForceOptimizer
        case OptimizerName.RANDOM:
            klass = RandomOptimizer
        case OptimizerName.UNIFORM:
            klass = UniformOptimizer
        case _:
            raise ValueError(f"Unknown optimizer name: {name}")

    return common.correct_kwargs(core.evaluate_corpus)(
        klass, corpus=corpus, adaptor=adaptor, **kwargs
    )
