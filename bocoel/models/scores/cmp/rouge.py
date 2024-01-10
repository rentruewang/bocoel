from collections.abc import Sequence

from bocoel.models.lms import LanguageModel
from bocoel.models.scores.interfaces import CmpScore


class RougeScore(CmpScore):
    def __init__(self, problem: str, answers: str, lm: LanguageModel) -> None:
        # Optional dependency.
        from rouge import Rouge

        self._problem = problem
        self._answers = answers
        self._lm = lm
        self._rouge = Rouge()

    def compare(
        self, generated: Sequence[str], reference: Sequence[Sequence[str]]
    ) -> Sequence[float]:
        return [
            self._rouge.get_scores(gen, ans) for gen, ans in zip(generated, reference)
        ]
