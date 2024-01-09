from collections.abc import Sequence

from rouge import Rouge

from bocoel.models.lms import LanguageModel

from .comparisons import CmpScore


class RougeScore(CmpScore):
    def __init__(self, problem: str, answer: str, lm: LanguageModel) -> None:
        self._problem = problem
        self._answer = answer
        self._lm = lm
        self._rouge = Rouge()

    def compare(
        self, generated: Sequence[str], reference: Sequence[str]
    ) -> Sequence[float]:
        return [
            self._rouge.get_scores(gen, ans) for gen, ans in zip(generated, reference)
        ]
