import pytest

import bocoel
from tests import utils
from tests.corpora import factories as corpus_factories
from tests.models.adaptors import factories as adaptor_factories
from tests.models.lms import factories as lm_factories

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
def test_init_optimizer(device: str) -> None:
    corpus = corpus_factories.corpus(device=device)
    lm = lm_factories.logits_lm(device=device)
    adaptor = adaptor_factories.sacre_bleu_eval(lm=lm)

    _ = factories.kmeans_optim(corpus, lm, adaptor)


@pytest.mark.parametrize("device", utils.torch_devices())
def test_optimize(
    device: str,
) -> None:
    corpus = corpus_factories.corpus(device=device)
    lm = lm_factories.logits_lm(device=device)
    adaptor = adaptor_factories.sacre_bleu_eval(lm=lm)
    optimizer = factories.kmeans_optim(corpus, lm, adaptor)

    bocoel.bocoel(optimizer=optimizer, iterations=15)
