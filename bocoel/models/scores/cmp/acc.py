from collections.abc import Sequence

from bocoel.models.lms import LanguageModel
from bocoel.models.scores.interfaces import CmpScore


class AccuracyScore(CmpScore):
    def __init__(self, problem: str, answer: str, lm: LanguageModel) -> None:
        self._problem = problem
        self._answer = answer
        self._lm = lm

    def compare(
        self, generated: Sequence[str], reference: Sequence[str]
    ) -> Sequence[float]:
        # TODO: Maybe handle special sequences?
        return [
            gen.lower().strip() == ans.lower().strip()
            for gen, ans in zip(generated, reference)
        ]
