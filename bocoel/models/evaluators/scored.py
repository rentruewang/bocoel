from collections.abc import Sequence

from numpy.typing import NDArray

from bocoel.corpora import Corpus
from bocoel.models import utils
from bocoel.models.evaluators.interfaces import CorpusEvaluator
from bocoel.models.scores import Score


class ScoredEvaluator(CorpusEvaluator):
    def __init__(self, corpus: Corpus, score: Score) -> None:
        self._corpus = corpus
        self._score = score

    def evaluate(self, indices: Sequence[int] | NDArray) -> Sequence[float] | NDArray:
        return utils.evaluate_on_corpus(
            score=self._score, corpus=self._corpus, indices=indices
        )
