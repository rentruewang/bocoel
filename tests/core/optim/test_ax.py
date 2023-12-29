import pytest
from pytest import FixtureRequest

from bocoel import AxServiceOptimizer, ComposedCore, ComposedCorpus, Corpus, Optimizer
from tests import utils
from tests.corpora import test_corpus
from tests.models.evaluators import test_bleu
from tests.models.lms import test_huggingface


def optim(corpus: Corpus) -> Optimizer:
    return AxServiceOptimizer(corpus)


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

    # FIXME: Sobol has 5 iterations.
    # This should be changed when the hardcode is removed.
    for _ in range(5):
        _ = core.step()
