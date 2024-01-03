import pytest

import bocoel
from bocoel import BleuEvaluator, Evaluator
from tests import utils
from tests.corpora import test_corpus
from tests.models.lms import test_huggingface


def bleu(device: str) -> Evaluator:
    return BleuEvaluator(
        problem="question", answer="answer", lm=test_huggingface.lm(device=device)
    )


@pytest.mark.parametrize("device", utils.torch_devices())
def test_bleu_eval(device: str) -> None:
    bleu_eval = bleu(device=device)
    corpus = test_corpus.corpus(device=device)

    results = bocoel.evaluate_on_corpus(bleu_eval, corpus, [0, 1])
    assert len(results) == 2
    assert all(0 <= r <= 1 for r in results), {"results": results, "corpus": corpus}
