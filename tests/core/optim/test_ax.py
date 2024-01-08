import pytest

import bocoel
from tests import utils
from tests.corpora import factories as corpus_factories
from tests.models.evaluators import factories as evaluator_factories
from tests.models.lms import factories as lm_factories

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
def test_init_optimizer(device: str) -> None:
    corpus = corpus_factories.corpus(device=device)
    evaluator = evaluator_factories.bleu(device=device)

    _ = factories.ax_optim(corpus, evaluator, device)


@pytest.mark.parametrize("device", utils.torch_devices())
def test_optimize(device: str) -> None:
    corpus = corpus_factories.corpus(device=device)
    lm = lm_factories.lm(device=device)
    evaluator = evaluator_factories.bleu(device=device)
    optimizer = factories.ax_optim(corpus, evaluator, device)

    bocoel.bocoel(optimizer=optimizer, iterations=5)
