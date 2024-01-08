import pytest

import bocoel
from tests import utils
from tests.corpora import factories as corpus_factories

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
def test_bleu_eval(device: str) -> None:
    bleu_eval = factories.bleu(device=device)
    corpus = corpus_factories.corpus(device=device)

    results = bocoel.evaluate_on_corpus(bleu_eval, corpus, [0, 1])
    assert len(results) == 2
    assert all(0 <= r <= 1 for r in results), {"results": results, "corpus": corpus}
