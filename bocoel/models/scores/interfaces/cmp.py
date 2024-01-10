import abc
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from numpy.typing import NDArray

from .scores import LanguageModelScore


class CmpScore(LanguageModelScore, Protocol):
    _problem: str
    _answers: str

    def compute(self, items: Mapping[str, Sequence[Any]]) -> Sequence[float] | NDArray:
        problems: Sequence[Any] = items[self._problem]
        answers = items[self._answers]
        generated = self._lm.generate(problems)
        return self.compare(generated=generated, references=answers)

    def compare(
        self, generated: Sequence[str], references: Sequence[Sequence[Any]]
    ) -> Sequence[float]:
        return [
            self.compare_one(generated=gen, references=ref)
            for gen, ref in zip(generated, references)
        ]

    @abc.abstractmethod
    def compare_one(self, generated: str, references: Sequence[Any]) -> float:
        ...
