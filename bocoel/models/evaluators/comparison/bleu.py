import typing
from collections.abc import Sequence

from nltk.translate import bleu_score
from numpy.typing import NDArray

from .comparison import ComparisonEvaluator


class BleuEvaluator(ComparisonEvaluator):
    def __init__(self, problem: str, answer: str) -> None:
        self._problem_key = problem
        self._answer_key = answer

    @property
    def source(self) -> str:
        return self._problem_key

    @property
    def target(self) -> str:
        return self._answer_key

    def _evaluate(
        self, generated: Sequence[str], answers: Sequence[str]
    ) -> Sequence[float] | NDArray:
        return [
            typing.cast(float, bleu_score.sentence_bleu([ans.split()], gen.split()))
            for ans, gen in zip(answers, generated)
        ]
