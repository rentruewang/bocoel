import abc
from collections.abc import Sequence
from typing import Protocol

from numpy.typing import NDArray

from bocoel.corpora import Corpus, Index


class Evaluator(Protocol):
    @property
    @abc.abstractmethod
    def index(self) -> Index:
        ...

    @abc.abstractmethod
    def evaluate(self, indices: Sequence[int] | NDArray) -> Sequence[float] | NDArray:
        ...


class CorpusEvaluator(Evaluator, Protocol):
    _corpus: Corpus

    @property
    def index(self) -> Index:
        return self._corpus.index
