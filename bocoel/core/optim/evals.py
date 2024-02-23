from collections.abc import Sequence
from typing import Any

import numpy as np
import structlog
from numpy.typing import ArrayLike

from bocoel.core.optim.interfaces import Optimizer
from bocoel.corpora import Corpus
from bocoel.models import Adaptor

LOGGER = structlog.get_logger()


def evaluate_corpus(
    optim_class: type[Optimizer], corpus: Corpus, adaptor: Adaptor, **kwargs: Any
) -> Optimizer:
    def index_eval(idx: ArrayLike, /) -> Sequence[float]:
        indices = np.array(idx)

        if indices.ndim != 1:
            raise ValueError(
                f"Expected indices to have dimensions [batch]. Got {indices.shape}"
            )

        LOGGER.debug("Evaluating indices", indices=indices.shape)
        evaluated = adaptor.on_corpus(corpus=corpus, indices=indices)
        assert (
            evaluated.shape == indices.shape
        ), f"Expected {indices.shape}. Got {evaluated.shape}"
        return evaluated.tolist()

    return optim_class(index_eval, corpus.index, **kwargs)
