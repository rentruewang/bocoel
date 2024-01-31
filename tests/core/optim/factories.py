from bocoel import (
    AcquisitionFunc,
    Adaptor,
    AxServiceOptimizer,
    Corpus,
    GenerativeModel,
    KMeansOptimizer,
    KMedoidsOptimizer,
    Optimizer,
    Task,
    core,
)
from tests import utils


@utils.cache
def ax_optim(
    corpus: Corpus, lm: GenerativeModel, adaptor: Adaptor, device: str, workers: int
) -> Optimizer:
    return core.evaluate_corpus(
        AxServiceOptimizer,
        corpus=corpus,
        adaptor=adaptor,
        sobol_steps=5,
        device=device,
        acqf=AcquisitionFunc.UCB,
        task=Task.MAXIMIZE,
        workers=workers,
    )


@utils.cache
def kmeans_optim(corpus: Corpus, lm: GenerativeModel, adaptor: Adaptor) -> Optimizer:
    return core.evaluate_corpus(
        KMeansOptimizer,
        corpus=corpus,
        adaptor=adaptor,
        batch_size=64,
        embeddings=corpus.index.embeddings,
        model_kwargs={"n_clusters": 3, "n_init": "auto"},
    )


@utils.cache
def kmedoids_optim(corpus: Corpus, lm: GenerativeModel, adaptor: Adaptor) -> Optimizer:
    return core.evaluate_corpus(
        KMedoidsOptimizer,
        corpus=corpus,
        adaptor=adaptor,
        batch_size=64,
        embeddings=corpus.index.embeddings,
        model_kwargs={"n_clusters": 3},
    )
