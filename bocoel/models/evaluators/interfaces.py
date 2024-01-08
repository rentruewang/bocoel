import abc
from collections.abc import Mapping, Sequence
from typing import Protocol

from numpy.typing import NDArray

from bocoel.models.lms import LanguageModel


class Evaluator(Protocol):
    """
    Evaluator protocol is used for evaluation of a given data structure.
    """

    @abc.abstractmethod
    def evaluate(self, items: Mapping[str, Sequence[str]]) -> Sequence[float] | NDArray:
        """
        Evaluate the language model on the given items.
        """

        ...


class LanguageModelEvaluator(Evaluator, Protocol):
    _lm: LanguageModel
    """
    The language model used by the evaluator.

    This method is private because it is only used by the evaluator itself.
    """
