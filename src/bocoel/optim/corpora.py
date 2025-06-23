# Copyright (c) BoCoEL Authors - All Rights Reserved

import dataclasses as dcls

import numpy as np
import structlog
from numpy.typing import ArrayLike, NDArray

from bocoel import Adaptor
from bocoel.corpora import Corpus

from .interfaces import IndexEvaluator

__all__ = ["CorpusEvaluator"]

LOGGER = structlog.get_logger()


@dcls.dataclass
class CorpusEvaluator(IndexEvaluator):
    """
    Evaluates the corpus with the given adaptor.
    """

    corpus: Corpus
    adaptor: Adaptor

    def __call__(self, idx: ArrayLike, /) -> NDArray:
        """
        Evaluates the given batched query.

        Parameters:
            idx: The indices to evaluate.

        Returns:
            The results of the query. Must be in the same order as the query.
        """

        indices = np.array(idx)

        if indices.ndim != 1:
            raise ValueError(
                f"Expected indices to have dimensions [batch]. Got {indices.shape}"
            )

        LOGGER.debug("Evaluating indices", indices=indices.shape)
        evaluated = self.adaptor.on_corpus(corpus=self.corpus, indices=indices)
        assert (
            evaluated.shape == indices.shape
        ), f"Expected {indices.shape}. Got {evaluated.shape}"
        return evaluated
