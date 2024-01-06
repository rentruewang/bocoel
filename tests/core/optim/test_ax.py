import pytest

import bocoel
from bocoel import AxServiceOptimizer, Corpus, Evaluator, Optimizer
from tests import utils
from tests.corpora import test_corpus
from tests.models.evaluators import test_bleu
from tests.models.lms import test_huggingface


def optim(corpus: Corpus, evaluator: Evaluator, device: str) -> Optimizer:
    steps = [
        {"model": "sobol", "num_trials": 5},
        {"model": "gpmes", "num_trials": 5, "model_kwargs": {"torch_device": device}},
    ]
    return AxServiceOptimizer.from_steps(corpus, evaluator, steps=steps)


@pytest.mark.parametrize("device", utils.torch_devices())
def test_init_optimizer(device: str) -> None:
    corpus = test_corpus.corpus(device=device)
    evaluator = test_bleu.bleu(device=device)

    _ = optim(corpus, evaluator, device)


@pytest.mark.parametrize("device", utils.torch_devices())
def test_optimize(device: str) -> None:
    corpus = test_corpus.corpus(device=device)
    lm = test_huggingface.lm(device=device)
    evaluator = test_bleu.bleu(device=device)
    optimizer = optim(corpus, evaluator, device)

    bocoel.bocoel(optimizer=optimizer, iterations=5)
