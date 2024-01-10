from collections.abc import Sequence
from typing import Any

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

    def compare_one(self, generated: str, references: Sequence[Any]) -> float:
        return self._rouge.get_scores(generated, references)
