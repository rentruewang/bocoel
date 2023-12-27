import pytest

from bocoel import BleuEvaluator, Evaluator
from tests import utils
from tests.corpora import test_corpus
from tests.models.lms import test_huggingface


def bleu() -> Evaluator:
    return BleuEvaluator(problem="question", answer="answer")


@pytest.fixture
def bleu_fix() -> Evaluator:
    return bleu()


@pytest.mark.parametrize("device", utils.devices())
def test_bleu_eval(bleu_fix: Evaluator, device: str) -> None:
    lm = test_huggingface.lm(device=device)
    corpus = test_corpus.corpus(device=device)

    results = bleu_fix.evaluate(lm=lm, corpus=corpus, indices=[0])
    assert len(results) == 1
    assert 0 <= results[0] <= 1, {
        "results": results,
        "lm": lm,
        "corpus": corpus,
    }
