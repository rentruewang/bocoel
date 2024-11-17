# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import numpy as np
import structlog
from numpy.typing import ArrayLike, NDArray

from bocoel.core.optim.interfaces import IndexEvaluator
from bocoel.corpora import Corpus
from bocoel.models import Adaptor

LOGGER = structlog.get_logger()


class CorpusEvaluator(IndexEvaluator):
    """
    Evaluates the corpus with the given adaptor.
    """

    def __init__(self, corpus: Corpus, adaptor: Adaptor) -> None:
        self.corpus = corpus
        self.adaptor = adaptor

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
