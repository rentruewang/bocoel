from bocoel import (
    AcquisitionFunc,
    AxServiceOptimizer,
    Corpus,
    Evaluator,
    KMeansOptimizer,
    LanguageModel,
    Optimizer,
    Task,
)


def ax_optim(
    corpus: Corpus, lm: LanguageModel, evaluator: Evaluator, device: str
) -> Optimizer:
    return AxServiceOptimizer.evaluate_corpus(
        corpus=corpus,
        lm=lm,
        evaluator=evaluator,
        sobol_steps=5,
        device=device,
        acqf=AcquisitionFunc.UPPER_CONFIDENCE_BOUND,
        task=Task.MAXIMIZE,
    )


def kmeans_optim(corpus: Corpus, lm: LanguageModel, evaluator: Evaluator) -> Optimizer:
    return KMeansOptimizer.evaluate_corpus(
        corpus=corpus, lm=lm, evaluator=evaluator, n_clusters=3
    )
