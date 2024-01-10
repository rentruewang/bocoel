import abc
from collections.abc import Mapping, Sequence
from typing import Protocol

from numpy.typing import NDArray

from .scores import LanguageModelScore


class CmpScore(LanguageModelScore, Protocol):
    _problem: str
    _answer: str

    def compute(self, items: Mapping[str, Sequence[str]]) -> Sequence[float] | NDArray:
        problems = items[self._problem]
        answers = items[self._answer]
        generated = self._lm.generate(problems)
        return self.compare(generated=generated, reference=answers)

    @abc.abstractmethod
    def compare(
        self, generated: Sequence[str], reference: Sequence[str]
    ) -> Sequence[float]:
        ...
