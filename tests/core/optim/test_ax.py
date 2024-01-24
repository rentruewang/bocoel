import pytest

import bocoel
from tests import utils
from tests.corpora import factories as corpus_factories
from tests.models.adaptors import factories as adaptor_factories
from tests.models.lms import factories as lm_factories

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_init_optimizer(device: str, workers: int) -> None:
    corpus = corpus_factories.corpus(device=device)
    lm = lm_factories.logits_lm(device=device)
    adaptor = adaptor_factories.sacre_bleu_eval()

    _ = factories.ax_optim(corpus, lm, adaptor, device=device, workers=workers)


@pytest.mark.parametrize("device", utils.torch_devices())
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_optimize(device: str, workers: int) -> None:
    corpus = corpus_factories.corpus(device=device)
    lm = lm_factories.logits_lm(device=device)
    adaptor = adaptor_factories.sacre_bleu_eval()
    optimizer = factories.ax_optim(corpus, lm, adaptor, device=device, workers=workers)

    bocoel.bocoel(optimizer=optimizer, iterations=10)
