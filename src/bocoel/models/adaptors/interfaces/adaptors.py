# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import structlog
from numpy.typing import ArrayLike, NDArray

from bocoel import common
from bocoel.corpora import Corpus, Storage

LOGGER = structlog.get_logger()


class Adaptor(Protocol):
    """
    Adaptors are the glue between scores and the corpus.
    It is designed to handle running a particular score on a particular corpus / dataset.
    """

    def __repr__(self) -> str:
        name = common.remove_base_suffix(self, Adaptor)
        return f"{name}()"

    @abc.abstractmethod
    def evaluate(self, data: Mapping[str, Sequence[Any]]) -> Sequence[float] | NDArray:
        """
        Evaluate a particular set of entries with a language model.
        Returns a list of scores, one for each entry, in the same order.

        Parameters:
            data: A mapping from column names to the data in that column.

        Returns:
            The scores for each entry. Scores must be floating point numbers.
        """

        ...

    def on_storage(self, storage: Storage, indices: ArrayLike) -> NDArray:
        """
        Evaluate a particular set of indices on a storage.
        Given indices and a storage,
        this method will extract the corresponding entries from the storage,
        and evaluate them with `Adaptor.evaluate`.

        Parameters:
            storage: The storage to evaluate.
            indices: The indices to evaluate.

        Returns:
            The scores for each entry. The shape must be the same as the indices.
        """

        indices = np.array(indices).astype("i")

        # Reshape the indices into 1D to evaluate.
        indices_shape = indices.shape
        indices = indices.ravel()

        items = storage[indices.tolist()]
        result = np.array(self.evaluate(data=items))

        # Reshape back.
        return result.reshape(indices_shape)

    def on_corpus(self, corpus: Corpus, indices: ArrayLike) -> NDArray:
        """
        Evaluate a particular set of indices on a corpus.
        A convenience wrapper around `Adaptor.on_storage`.

        Parameters:
            corpus: The corpus to evaluate.
            indices: The indices to evaluate.

        Returns:
            The scores for each entry. The shape must be the same as the indices.
        """

        return self.on_storage(storage=corpus.storage, indices=indices)
