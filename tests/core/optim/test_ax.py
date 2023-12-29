import pytest

from bocoel import AxServiceOptimizer, ComposedCore, Corpus, GenStepDict, Optimizer
from tests import utils
from tests.corpora import test_corpus
from tests.models.evaluators import test_bleu
from tests.models.lms import test_huggingface


def optim(corpus: Corpus) -> Optimizer:
    steps: list[GenStepDict] = [
        {"model": "sobol", "num_trials": 5},
        {"model": "gpmes", "num_trials": 5},
    ]
    return AxServiceOptimizer.from_steps(corpus, steps)


@pytest.mark.parametrize("device", utils.torch_devices())
def test_init_optimizer(device: str) -> None:
    corpus = test_corpus.corpus(device=device)
    _ = optim(corpus)


@pytest.mark.parametrize("device", utils.torch_devices())
def test_optimize(device: str) -> None:
    corpus = test_corpus.corpus(device=device)
    lm = test_huggingface.lm(device=device)
    evaluator = test_bleu.bleu()
    optimizer = optim(corpus)

    core = ComposedCore(corpus, lm, evaluator, optimizer)

    core.run(iterations=5)
