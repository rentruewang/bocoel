import pytest

import bocoel
from tests import utils
from tests.corpora import factories as corpus_factories
from tests.models.evaluators import factories as eval_factories
from tests.models.lms import factories as lm_factories

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
def test_init_optimizer(device: str) -> None:
    corpus = corpus_factories.corpus(device=device)
    lm = lm_factories.lm(device=device)
    evaluator = eval_factories.sacre_bleu_eval()

    _ = factories.kmeans_optim(corpus, lm, evaluator)


@pytest.mark.parametrize("device", utils.torch_devices())
def test_optimize(device: str) -> None:
    corpus = corpus_factories.corpus(device=device)
    lm = lm_factories.lm(device=device)
    evaluator = eval_factories.sacre_bleu_eval()
    optimizer = factories.kmeans_optim(corpus, lm, evaluator)

    bocoel.bocoel(optimizer=optimizer, iterations=15)
