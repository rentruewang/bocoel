import abc
import typing
from typing import Protocol, Sequence

from numpy.typing import NDArray

from bocoel.corpora import Corpus, Storage

from .lms import LanguageModel


class Evaluator(Protocol):
    """
    Evaluator protocol is used to evaluate the language model on a given corpus.
    """

    @typing.overload
    def evaluate(self, lm: LanguageModel, corpus: Corpus, indices: int) -> float:
        ...

    @typing.overload
    def evaluate(
        self, lm: LanguageModel, corpus: Corpus, indices: Sequence[int] | NDArray
    ) -> Sequence[float]:
        ...

    def evaluate(
        self, lm: LanguageModel, corpus: Corpus, indices: int | Sequence[int] | NDArray
    ) -> float | Sequence[float]:
        # FIXME: Using an if-else pattern to pack/unpack. Should remove in the future.

        pack = isinstance(indices, int)

        if pack:
            idxs = [indices]
        else:
            idxs = indices

        evaluation = self._evaluate(lm=lm, store=corpus.storage, indices=idxs)

        if pack:
            return evaluation[0]
        else:
            return evaluation

    @abc.abstractmethod
    def _evaluate(
        self, lm: LanguageModel, store: Storage, indices: Sequence[int] | NDArray
    ) -> Sequence[float]:
        ...
