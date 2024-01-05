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
        """
        Evaluate the language model on the given items.
        """

        ...


# TODO:
# Further modularize reusable components
# when expanding on numbers of supporting functionalities.
class LanguageModelEvaluator(Evaluator, Protocol):
    @property
    @abc.abstractmethod
    def _lm(self) -> LanguageModel:
        """
        The language model used by the evaluator.

        This method is private because it is only used by the evaluator itself.
        """

        ...
