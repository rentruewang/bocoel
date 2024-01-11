import pytest

from tests import utils
from tests.corpora import factories as corpus_factories
from tests.models.lms import factories as lm_factories

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
def test_bleu_eval(device: str) -> None:
    bleu_eval = factories.bleu()
    corpus = corpus_factories.corpus(device=device)
    lm = lm_factories.lm(device=device)

    results = bleu_eval.on_corpus(corpus=corpus, lm=lm, indices=[0, 1])
    assert len(results) == 2
    assert all(0 <= r <= 1 for r in results), {"results": results, "corpus": corpus}
