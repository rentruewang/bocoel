import pytest

from bocoel import Manager, SbertEmbedder
from tests import utils
from tests.corpora import factories as corpus_factories
from tests.models.adaptors import factories as adaptor_factories
from tests.models.lms import factories as lm_factories

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
@pytest.mark.parametrize(
    "score",
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
def test_init_optimizer(device: str, score: str) -> None:
    embedder = SbertEmbedder(device=device)
    corpus = corpus_factories.corpus(embedder=embedder)
    lm = lm_factories.generative_lm(device=device)
    adaptor = adaptor_factories.bigbench_adaptor(name=score, lm=lm)

    _ = factories.kmedoids_optim(corpus, lm, adaptor)


@pytest.mark.parametrize("device", utils.torch_devices())
@pytest.mark.parametrize(
    "score",
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
def test_optimize(device: str, score: str) -> None:
    embedder = SbertEmbedder(device=device)
    corpus = corpus_factories.corpus(embedder=embedder)
    lm = lm_factories.generative_lm(device=device)
    adaptor = adaptor_factories.bigbench_adaptor(name=score, lm=lm)
    optimizer = factories.kmedoids_optim(corpus, lm, adaptor)

    Manager().run(
        optimizer=optimizer,
        embedder=embedder,
        corpus=corpus,
        model=lm,
        adaptor=adaptor,
        steps=15,
    )
