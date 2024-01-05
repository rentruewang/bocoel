import typing
from collections.abc import Mapping, Sequence

from nltk.translate import bleu_score
from numpy.typing import NDArray

from bocoel.models.evaluators.interfaces import LanguageModel, LanguageModelEvaluator


class BleuEvaluator(LanguageModelEvaluator):
    def __init__(self, problem: str, answer: str, lm: LanguageModel) -> None:
        self._problem = problem
        self._answer = answer
        self._lm = lm

    @property
    def _lm(self) -> LanguageModel:
        return self._lm

    def evaluate(self, items: Mapping[str, Sequence[str]]) -> Sequence[float] | NDArray:
        problems = items[self._problem]
        answers: Sequence[str] = items[self._answer]
        generated = self._lm.generate(problems)
        return [
            typing.cast(float, bleu_score.sentence_bleu([ans.split()], gen.split()))
            for ans, gen in zip(answers, generated)
        ]
