from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import structlog
from numpy import random
from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.core.evals import State
from bocoel.core.optim import utils as optim_utils
from bocoel.core.optim.interfaces import Optimizer, Task
from bocoel.corpora import Index, SearchResult

LOGGER = structlog.get_logger()


class RandomOptimizer(Optimizer):
    def __init__(
        self,
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
        *,
        samples: int,
    ) -> None:
        LOGGER.info("Instantiating RandomOptimizer", samples=samples)

        self._index = index
        self._evaluate_fn = evaluate_fn
        self._samples = samples

    @property
    def task(self) -> Task:
        return Task.EXPLORE

    @property
    def terminate(self) -> bool:
        return True

    def step(self) -> Sequence[State]:
        # Cache the embeddings to save possible computation.
        embeddings = self._index.embeddings
        minimum = np.min(embeddings, axis=0)
        maximum = np.max(embeddings, axis=0)

        samples = random.random([self._samples, self._index.dims])
        samples *= maximum - minimum
        samples += minimum

        LOGGER.debug(
            "Generated samples", samples=samples, minimum=minimum, maximum=maximum
        )

        return optim_utils.evaluate_index(
            query=samples, index=self._index, evaluate_fn=self._evaluate_fn
        )

    @classmethod
    def from_index(
        cls,
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
        **kwargs: Any,
    ) -> Self:
        return cls(index=index, evaluate_fn=evaluate_fn, **kwargs)
