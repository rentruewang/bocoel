import abc
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from numpy.typing import NDArray

from bocoel.corpora import Corpus, Storage
from bocoel.models.lms import LanguageModel

from . import utils


class Evaluator(Protocol):
    @abc.abstractmethod
    def evaluate(
        self, data: Mapping[str, Sequence[Any]], lm: LanguageModel
    ) -> Sequence[float] | NDArray:
        ...

    def on_storage(
        self, storage: Storage, lm: LanguageModel, indices: Sequence[int] | NDArray
    ) -> Sequence[float] | NDArray:
        items = [storage[idx] for idx in indices]

        collated = utils.collate(items)

        return self.evaluate(data=collated, lm=lm)

    def on_corpus(
        self, corpus: Corpus, lm: LanguageModel, indices: Sequence[int] | NDArray
    ) -> Sequence[float] | NDArray:
        return self.on_storage(storage=corpus.storage, lm=lm, indices=indices)
