import pytest

import bocoel
from tests import utils
from tests.corpora import factories as corpora_factories
from tests.models.evaluators import test_bleu
from tests.models.lms import test_huggingface

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
def test_init_optimizer(device: str) -> None:
    corpus = corpora_factories.corpus(device=device)
    evaluator = test_bleu.bleu(device=device)

    _ = factories.kmeans_optim(corpus, evaluator)


@pytest.mark.parametrize("device", utils.torch_devices())
def test_optimize(device: str) -> None:
    corpus = corpora_factories.corpus(device=device)
    lm = test_huggingface.lm(device=device)
    evaluator = test_bleu.bleu(device=device)
    optimizer = factories.kmeans_optim(corpus, evaluator)

    bocoel.bocoel(optimizer=optimizer, iterations=5)
