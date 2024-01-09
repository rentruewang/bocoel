import pytest

import bocoel
from tests import utils
from tests.corpora import factories as corpus_factories
from tests.models.scores import factories as score_factories

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
def test_init_optimizer(device: str) -> None:
    corpus = corpus_factories.corpus(device=device)
    score = score_factories.bleu(device=device)

    _ = factories.ax_optim(corpus, score)


@pytest.mark.parametrize("device", utils.torch_devices())
def test_optimize(device: str) -> None:
    corpus = corpus_factories.corpus(device=device)
    score = score_factories.bleu(device=device)
    optimizer = factories.ax_optim(corpus, score)

    bocoel.bocoel(optimizer=optimizer, iterations=10)
