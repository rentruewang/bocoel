from bocoel import (
    AxServiceOptimizer,
    Corpus,
    Evaluator,
    KMeansOptimizer,
    LanguageModel,
    Optimizer,
)


def ax_optim(corpus: Corpus, lm: LanguageModel, evaluator: Evaluator) -> Optimizer:
    steps = [
        {"model": "sobol", "num_trials": 5},
        {"model": "modular", "num_trials": -1},
    ]
    return AxServiceOptimizer.evaluate_corpus(
        corpus=corpus, lm=lm, evaluator=evaluator, steps=steps
    )


def kmeans_optim(corpus: Corpus, lm: LanguageModel, evaluator: Evaluator) -> Optimizer:
    return KMeansOptimizer.evaluate_corpus(
        corpus=corpus, lm=lm, evaluator=evaluator, n_clusters=3
    )
