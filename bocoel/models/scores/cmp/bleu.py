import typing
from collections.abc import Sequence

from bocoel.models.lms import LanguageModel
from bocoel.models.scores.interfaces import CmpScore


class BleuScore(CmpScore):
    def __init__(self, problem: str, answers: Sequence[str], lm: LanguageModel) -> None:
        # Optional dependency.
        from nltk.translate import bleu_score

        self._problem = problem
        self._answers = answers
        self._lm = lm

        self._bleu = bleu_score.sentence_bleu

    # TODO: Improve performance.
    def compare(
        self, generated: Sequence[str], reference: Sequence[Sequence[str]]
    ) -> Sequence[float]:
        return [
            typing.cast(float, self._bleu([a.split() for a in ans], gen.split()))
            for ans, gen in zip(reference, generated)
        ]
