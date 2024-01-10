from collections.abc import Sequence

from bocoel.models.lms import LanguageModel
from bocoel.models.scores.interfaces import CmpScore


class AccuracyScore(CmpScore):
    def __init__(self, problem: str, answers: Sequence[str], lm: LanguageModel) -> None:
        self._problem = problem
        self._answers = answers
        self._lm = lm

    def compare(
        self, generated: Sequence[str], reference: Sequence[Sequence[str]]
    ) -> Sequence[float]:
        # TODO: Maybe handle special sequences?
        return [
            _cleanup(gen) in [_cleanup(a) for a in ans]
            for gen, ans in zip(generated, reference)
        ]


def _cleanup(string: str) -> str:
    return string.lower().strip()
