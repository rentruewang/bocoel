import functools

from bocoel import (
    AcquisitionFunc,
    AxServiceOptimizer,
    Corpus,
    Evaluator,
    KMeansOptimizer,
    KMedoidsOptimizer,
    LanguageModel,
    Optimizer,
    Task,
)


@functools.cache
def ax_optim(
    corpus: Corpus, lm: LanguageModel, evaluator: Evaluator, device: str, workers: int
) -> Optimizer:
    return AxServiceOptimizer.evaluate_corpus(
        corpus=corpus,
        lm=lm,
        evaluator=evaluator,
        sobol_steps=5,
        device=device,
        acqf=AcquisitionFunc.UCB,
        task=Task.MAXIMIZE,
        workers=workers,
    )


@functools.cache
def kmeans_optim(corpus: Corpus, lm: LanguageModel, evaluator: Evaluator) -> Optimizer:
    return KMeansOptimizer.evaluate_corpus(
        corpus=corpus,
        lm=lm,
        evaluator=evaluator,
        model_kwargs={"n_clusters": 3, "n_init": "auto"},
    )


@functools.cache
def kmedoids_optim(
    corpus: Corpus, lm: LanguageModel, evaluator: Evaluator
) -> Optimizer:
    return KMedoidsOptimizer.evaluate_corpus(
        corpus=corpus,
        lm=lm,
        evaluator=evaluator,
        model_kwargs={"n_clusters": 3},
    )
