from collections.abc import Sequence
from typing import Any

from bocoel.models.lms import LanguageModel
from bocoel.models.scores.interfaces import CmpScore


class BleuScore(CmpScore):
    def __init__(self, problem: str, answers: str, lm: LanguageModel) -> None:
        # Optional dependency.
        from nltk.translate import bleu_score

        self._problem = problem
        self._answers = answers
        self._lm = lm

        self._bleu = bleu_score.sentence_bleu

    def compare_one(self, generated: str, references: Sequence[Any]) -> float:
        return self._bleu([ref.split() for ref in references], generated.split())
