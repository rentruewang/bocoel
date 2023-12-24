from typing import Mapping, Sequence

from nltk.translate import bleu_score
from numpy.typing import NDArray

from bocoel.corpora import Storage
from bocoel.models.interfaces import Evaluator, LanguageModel


class Bleu(Evaluator):
    def eval(
        self,
        lm: LanguageModel,
        store: Storage,
        indices: Sequence[int] | NDArray,
        /,
        *keys: str,
    ) -> Sequence[float]:
        problem_key, answer_key = keys

        # FIXME: Only allows one correct answer in BLEU as of right now.

        problems = [store[idx][problem_key] for idx in indices]
        answers = [store[idx][answer_key] for idx in indices]
        generated = lm.generate(problems)

        return [
            bleu_score.sentence_bleu([ans.split()], gen.split())
            for ans, gen in zip(answers, generated)
        ]

    def _validate_inputs(self, *keys: str) -> None:
        if len(keys) != 2:
            raise ValueError(f"Bleu evaluator requires exactly two keys. Got: {keys}")
