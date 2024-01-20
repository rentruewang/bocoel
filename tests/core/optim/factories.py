from bocoel import (
    AcquisitionFunc,
    Adaptor,
    AxServiceOptimizer,
    Corpus,
    KMeansOptimizer,
    KMedoidsOptimizer,
    LanguageModel,
    Optimizer,
    Task,
)
from tests import utils


@utils.cache
def ax_optim(
    corpus: Corpus, lm: LanguageModel, adaptor: Adaptor, device: str, workers: int
) -> Optimizer:
    return AxServiceOptimizer.evaluate_corpus(
        corpus=corpus,
        lm=lm,
        adaptor=adaptor,
        sobol_steps=5,
        device=device,
        acqf=AcquisitionFunc.UCB,
        task=Task.MAXIMIZE,
        workers=workers,
    )


@utils.cache
def kmeans_optim(corpus: Corpus, lm: LanguageModel, adaptor: Adaptor) -> Optimizer:
    return KMeansOptimizer.evaluate_corpus(
        corpus=corpus,
        lm=lm,
        adaptor=adaptor,
        model_kwargs={"n_clusters": 3, "n_init": "auto"},
    )


@utils.cache
def kmedoids_optim(corpus: Corpus, lm: LanguageModel, adaptor: Adaptor) -> Optimizer:
    return KMedoidsOptimizer.evaluate_corpus(
        corpus=corpus,
        lm=lm,
        adaptor=adaptor,
        model_kwargs={"n_clusters": 3},
    )
