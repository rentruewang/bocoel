from collections.abc import Sequence
from typing import Any

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
        LOGGER.debug("Evaluating indices", indices=idx)
        evaluated = adaptor.on_corpus(corpus=corpus, indices=idx)
        assert (
            evaluated.ndim == 1
        ), f"Should have the dimensions [batch]. Got {evaluated.shape}"
        return evaluated.tolist()

    return optim_class(index_eval=index_eval, boundary=corpus.index.boundary, **kwargs)
