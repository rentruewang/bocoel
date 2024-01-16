import abc
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bocoel.corpora import Corpus, Storage
from bocoel.models.evaluators import utils
from bocoel.models.lms import LanguageModel


class Evaluator(Protocol):
    """
    Evaluators are the adaptors between scores, langauge models, and the corpus.
    It is designed to handle running a particular score on a particular corpus / dataset.
    """

    @abc.abstractmethod
    def evaluate(
        self, data: Mapping[str, Sequence[Any]], lm: LanguageModel
    ) -> Sequence[float] | NDArray:
        ...

    def on_storage(
        self, storage: Storage, lm: LanguageModel, indices: ArrayLike
    ) -> NDArray:
        indices = np.array(indices)

        # Reshape the indices into 1D to evaluate.
        indices_shape = indices.shape
        indices = indices.ravel()

        items = [storage[idx] for idx in indices.tolist()]
        collated = utils.collate(items)
        result = np.array(self.evaluate(data=collated, lm=lm))

        # Reshape back.
        return result.reshape(indices_shape)

    def on_corpus(
        self, corpus: Corpus, lm: LanguageModel, indices: ArrayLike
    ) -> NDArray:
        return self.on_storage(storage=corpus.storage, lm=lm, indices=indices)
