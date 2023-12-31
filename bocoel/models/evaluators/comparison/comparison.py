import abc
from collections.abc import Sequence

from numpy.typing import NDArray

from bocoel.corpora import Corpus
from bocoel.models.interfaces import Evaluator, LanguageModel


class ComparisonEvaluator(Evaluator):
    @property
    @abc.abstractmethod
    def source(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def target(self) -> str:
        ...

    @abc.abstractmethod
    def _evaluate(
        self, generated: Sequence[str], answers: Sequence[str]
    ) -> Sequence[float] | NDArray:
        ...

    def evaluate(
        self, lm: LanguageModel, corpus: Corpus, indices: Sequence[int] | NDArray
    ) -> Sequence[float] | NDArray:
        retrieved = [corpus.storage[idx] for idx in indices]
        problems = [r[self.source] for r in retrieved]
        answers = [r[self.target] for r in retrieved]
        generated = lm.generate(problems)

        return self._evaluate(generated=generated, answers=answers)
