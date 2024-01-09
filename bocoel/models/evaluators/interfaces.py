import abc
from collections.abc import Sequence
from typing import Protocol

from numpy.typing import NDArray

from bocoel.corpora import Corpus


class Evaluator(Protocol):
    corpus: Corpus

    @abc.abstractmethod
    def evaluate(self, indices: Sequence[int] | NDArray) -> Sequence[float] | NDArray:
        ...
