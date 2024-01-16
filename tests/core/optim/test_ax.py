import pytest

import bocoel
from tests import utils
from tests.corpora import factories as corpus_factories
from tests.models.evaluators import factories as eval_factories
from tests.models.lms import factories as lm_factories

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_init_optimizer(device: str, workers: int) -> None:
    corpus = corpus_factories.corpus(device=device)
    lm = lm_factories.lm(device=device)
    evaluator = eval_factories.sacre_bleu_eval()

    _ = factories.ax_optim(corpus, lm, evaluator, device=device, workers=workers)


@pytest.mark.parametrize("device", utils.torch_devices())
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_optimize(device: str, workers: int) -> None:
    corpus = corpus_factories.corpus(device=device)
    lm = lm_factories.lm(device=device)
    evaluator = eval_factories.sacre_bleu_eval()
    optimizer = factories.ax_optim(
        corpus, lm, evaluator, device=device, workers=workers
    )

    bocoel.bocoel(optimizer=optimizer, iterations=10)
