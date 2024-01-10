from collections.abc import Sequence
from typing import Any

from bocoel.models.lms import LanguageModel
from bocoel.models.scores.interfaces import CmpScore


class AccuracyScore(CmpScore):
    def __init__(self, problem: str, answers: str, lm: LanguageModel) -> None:
        self._problem = problem
        self._answers = answers
        self._lm = lm

    def compare_one(self, generated: str, references: Sequence[Any]) -> float:
        return float(_cleanup(generated) in [_cleanup(ref) for ref in references])


def _cleanup(string: str) -> str:
    return string.lower().strip()
