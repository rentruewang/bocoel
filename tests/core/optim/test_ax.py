import pytest
from ax.modelbridge.generation_strategy import GenerationStep
from ax.modelbridge.registry import Models

from bocoel import AxServiceOptimizer, ComposedCore, Corpus, Optimizer
from tests import utils
from tests.corpora import test_corpus
from tests.models.evaluators import test_bleu
from tests.models.lms import test_huggingface


def optim(corpus: Corpus) -> Optimizer:
    steps = [
        GenerationStep(
            model=Models.SOBOL,
            num_trials=5,
        ),
        GenerationStep(
            model=Models.GPMES,
            num_trials=-1,
        ),
    ]
    return AxServiceOptimizer.from_steps(corpus, steps)


@pytest.mark.parametrize("device", utils.devices())
def test_init_optimizer(device: str) -> None:
    corpus = test_corpus.corpus(device=device)
    _ = optim(corpus)


@pytest.mark.parametrize("device", utils.devices())
def test_optimize(device: str) -> None:
    corpus = test_corpus.corpus(device=device)
    lm = test_huggingface.lm(device=device)
    evaluator = test_bleu.bleu()
    optimizer = optim(corpus)

    core = ComposedCore(corpus, lm, evaluator, optimizer)

    core.run(iterations=5)
