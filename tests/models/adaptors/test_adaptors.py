import pytest

from bocoel import SbertEmbedder
from tests import utils
from tests.corpora import factories as corpus_factories
from tests.models.lms import factories as lm_factories

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
@pytest.mark.parametrize(
    "adaptor_name",
    [
        "sacre-bleu",
        "nltk-bleu",
        "rouge-1",
        "rouge-2",
        "rouge-l",
        "rouge-score-1",
        "rouge-score-2",
        "rouge-score-l",
        "exact-match",
    ],
)
def test_bigbench_adaptor_on_corpus(adaptor_name: str, device: str) -> None:
    embedder = SbertEmbedder(device=device)
    corpus = corpus_factories.corpus(embedder=embedder)
    lm = lm_factories.generative_lm(device=device)
    ev = factories.bigbench_adaptor(adaptor_name, lm=lm)

    results = ev.on_corpus(corpus=corpus, indices=[0, 1])
    assert len(results) == 2

    assert all(0 <= r <= 1 for r in results), {"results": results, "corpus": corpus}
