import abc
from collections.abc import Sequence

from numpy.typing import NDArray

from bocoel.corpora import Corpus
from bocoel.models.interfaces import Evaluator, LanguageModel


class LoneEvaluator(Evaluator):
    @property
    @abc.abstractmethod
    def source(self) -> str:
        ...

    @abc.abstractmethod
    def _evaluate(self, generated: Sequence[str]) -> Sequence[float] | NDArray:
        ...

    def evaluate(
        self, lm: LanguageModel, corpus: Corpus, indices: Sequence[int] | NDArray
    ) -> Sequence[float] | NDArray:
        retrieved = [corpus.storage[idx] for idx in indices]
        problems = [r[self.source] for r in retrieved]
        generated = lm.generate(problems)
        return self._evaluate(generated=generated)
