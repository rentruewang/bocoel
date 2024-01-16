import pytest

from tests import utils
from tests.corpora import factories as corpus_factories
from tests.models.lms import factories as lm_factories

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
@pytest.mark.parametrize(
    "evaluator_name",
    [
        "sacre_bleu",
        "nltk_bleu",
        "rouge-1",
        "rouge-2",
        "rouge-l",
        "rouge-score-1",
        "rouge-score-2",
        "rouge-score-l",
        "exact_match",
    ],
)
def test_evaluator_on_corpus(evaluator_name: str, device: str) -> None:
    corpus = corpus_factories.corpus(device=device)
    lm = lm_factories.lm(device=device)
    ev = factories.evaluator(evaluator_name)

    results = ev.on_corpus(corpus=corpus, lm=lm, indices=[0, 1])
    assert len(results) == 2

    assert all(0 <= r <= 1 for r in results), {"results": results, "corpus": corpus}
