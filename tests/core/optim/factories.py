from bocoel import (
    AcquisitionFunc,
    Adaptor,
    AxServiceOptimizer,
    Corpus,
    CorpusEvaluator,
    GenerativeModel,
    KMeansOptimizer,
    KMedoidsOptimizer,
    Optimizer,
    Task,
)
from tests import utils


@utils.cache
def ax_optim(
    corpus: Corpus, lm: GenerativeModel, adaptor: Adaptor, device: str, workers: int
) -> Optimizer:
    corpus_eval = CorpusEvaluator(corpus=corpus, adaptor=adaptor)
    return AxServiceOptimizer(
        index_eval=corpus_eval,
        index=corpus.index,
        sobol_steps=5,
        device=device,
        acqf=AcquisitionFunc.UCB,
        task=Task.MAXIMIZE,
        workers=workers,
    )


@utils.cache
def kmeans_optim(corpus: Corpus, lm: GenerativeModel, adaptor: Adaptor) -> Optimizer:
    corpus_eval = CorpusEvaluator(corpus=corpus, adaptor=adaptor)
    return KMeansOptimizer(
        index_eval=corpus_eval,
        index=corpus.index,
        batch_size=64,
        embeddings=corpus.index.data,
        model_kwargs={"n_clusters": 3, "n_init": "auto"},
    )


@utils.cache
def kmedoids_optim(corpus: Corpus, lm: GenerativeModel, adaptor: Adaptor) -> Optimizer:
    corpus_eval = CorpusEvaluator(corpus=corpus, adaptor=adaptor)
    return KMedoidsOptimizer(
        index_eval=corpus_eval,
        index=corpus.index,
        batch_size=64,
        embeddings=corpus.index.data,
        model_kwargs={"n_clusters": 3},
    )
