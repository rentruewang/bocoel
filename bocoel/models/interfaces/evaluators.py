import abc
import typing
from collections.abc import Sequence
from typing import Protocol

from numpy.typing import NDArray

from bocoel.corpora import Corpus, Storage

from .lms import LanguageModel


class Evaluator(Protocol):
    """
    Evaluator protocol is used to evaluate the language model on a given corpus.
    """

    @abc.abstractmethod
    def evaluate(
        self, lm: LanguageModel, corpus: Corpus, indices: Sequence[int] | NDArray
    ) -> Sequence[float] | NDArray:
        ...
