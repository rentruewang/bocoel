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
    steps = [
        {"model": "sobol", "num_trials": 5},
        {"model": "modular", "num_trials": -1},
    ]
    return AxServiceOptimizer.evaluate_corpus(
        corpus=corpus,
        lm=lm,
        evaluator=evaluator,
        sobol_steps=5,
        device=device,
        acqf=AcquisitionFunc.UPPER_CONFIDENCE_BOUND,
        Task=Task.MAXIMIZE,
    )


def kmeans_optim(corpus: Corpus, lm: LanguageModel, evaluator: Evaluator) -> Optimizer:
    return KMeansOptimizer.evaluate_corpus(
        corpus=corpus, lm=lm, evaluator=evaluator, n_clusters=3
    )
