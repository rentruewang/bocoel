import typing
from collections.abc import Sequence

from nltk.translate import bleu_score
from numpy.typing import NDArray

from bocoel.corpora import Storage
from bocoel.models.interfaces import Evaluator, LanguageModel


class BleuEvaluator(Evaluator):
    def __init__(self, problem: str, answer: str) -> None:
        self._problem_key = problem
        self._answer_key = answer

    def _evaluate(
        self, lm: LanguageModel, store: Storage, indices: Sequence[int] | NDArray
    ) -> Sequence[float]:
        # FIXME: Only allows one correct answer in BLEU as of right now.

        retrieved = [store[idx] for idx in indices]
        problems = [r[self._problem_key] for r in retrieved]
        answers = [r[self._answer_key] for r in retrieved]
        generated = lm.generate(problems)

        return [
            typing.cast(float, bleu_score.sentence_bleu([ans.split()], gen.split()))
            for ans, gen in zip(answers, generated)
        ]
