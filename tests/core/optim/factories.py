from bocoel import AxServiceOptimizer, Corpus, Evaluator, KMeansOptimizer, Optimizer


def ax_optim(corpus: Corpus, evaluator: Evaluator, device: str) -> Optimizer:
    steps = [
        {"model": "sobol", "num_trials": 5},
        {"model": "gpmes", "num_trials": 5, "model_kwargs": {"torch_device": device}},
    ]
    return AxServiceOptimizer.evaluate_corpus(
        corpus=corpus, evaluator=evaluator, steps=steps
    )


def kmeans_optim(corpus: Corpus, evaluator: Evaluator) -> Optimizer:
    return KMeansOptimizer.evaluate_corpus(corpus=corpus, evaluator=evaluator)
