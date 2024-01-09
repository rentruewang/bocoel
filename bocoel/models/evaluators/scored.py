from collections.abc import Sequence

from numpy.typing import NDArray

from bocoel.corpora import Corpus
from bocoel.models import utils
from bocoel.models.evaluators.interfaces import Evaluator
from bocoel.models.scores import Score


class ScoredEvaluator(Evaluator):
    def __init__(self, corpus: Corpus, score: Score) -> None:
        super().__init__()

        # Public attributes because they can be modified at anytime.
        self.corpus = corpus
        self.score = score

    def evaluate(self, indices: Sequence[int] | NDArray) -> Sequence[float] | NDArray:
        return utils.evaluate_on_corpus(
            score=self.score, corpus=self.corpus, indices=indices
        )
