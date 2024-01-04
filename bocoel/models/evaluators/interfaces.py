import abc
from collections.abc import Mapping, Sequence
from typing import Protocol

from numpy.typing import NDArray

from bocoel.models.lms import LanguageModel


class Evaluator(Protocol):
    """
    Evaluator protocol is used to evaluate the language model on a given corpus.
    """

    @abc.abstractmethod
    def evaluate(self, items: Mapping[str, Sequence[str]]) -> Sequence[float] | NDArray:
        ...


# TODO:
# Further modularize reusable components
# when expanding on numbers of supporting functionalities.
class LanguageModelEvaluator(Evaluator, Protocol):
    @property
    @abc.abstractmethod
    def lm(self) -> LanguageModel:
        ...
