import pytest

import bocoel
from tests import utils
from tests.corpora import factories as corpus_factories
from tests.models.lms import factories as lm_factories
from tests.models.scores import factories as score_factories

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
def test_init_optimizer(device: str) -> None:
    corpus = corpus_factories.corpus(device=device)
    score = score_factories.bleu(device=device)

    _ = factories.kmeans_optim(corpus, score)


@pytest.mark.parametrize("device", utils.torch_devices())
def test_optimize(device: str) -> None:
    corpus = corpus_factories.corpus(device=device)
    lm = lm_factories.lm(device=device)
    score = score_factories.bleu(device=device)
    optimizer = factories.kmeans_optim(corpus, score)

    bocoel.bocoel(optimizer=optimizer, iterations=15)
