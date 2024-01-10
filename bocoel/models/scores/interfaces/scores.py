import abc
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from numpy.typing import NDArray

from bocoel.models.lms import LanguageModel


class Score(Protocol):
    """
    Score protocol is used for evaluation of a given data structure.
    """

    @abc.abstractmethod
    def compute(self, items: Mapping[str, Sequence[Any]]) -> Sequence[float] | NDArray:
        """
        Compute the score on the given items.
        """

        ...


class LanguageModelScore(Score, Protocol):
    _lm: LanguageModel
    """
    The language model used in the evaluation.

    This method is private because it is only used by the instance itself.
    """
