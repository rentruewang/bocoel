from collections.abc import Mapping, Sequence

import structlog
from numpy.typing import NDArray

from bocoel.core.optim.evals import QueryEvaluator
from bocoel.core.optim.interfaces import Optimizer
from bocoel.core.optim.utils import BatchedGenerator
from bocoel.core.tasks import Task
from bocoel.corpora import Boundary

LOGGER = structlog.get_logger()


class BruteForceOptimizer(Optimizer):
    def __init__(
        self,
        query_eval: QueryEvaluator,
        boundary: Boundary,
        *,
        embeddings: Sequence[NDArray],
        batch_size: int,
    ) -> None:
        LOGGER.info("Instantiating CompleteOptimizer", samples=len(embeddings))

        self._query_eval = query_eval
        self._boundary = boundary
        self._generator = iter(BatchedGenerator(iter(embeddings), batch_size))

    @property
    def task(self) -> Task:
        return Task.EXPLORE

    @property
    def terminate(self) -> bool:
        return True

    def step(self) -> Mapping[int, float]:
        samples = next(self._generator)
        return self._query_eval(samples)
