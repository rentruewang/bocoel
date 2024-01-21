from collections.abc import Mapping

import structlog
from numpy import random
from numpy.typing import NDArray

from bocoel.core.optim.evals import QueryEvaluator
from bocoel.core.optim.interfaces import Optimizer, Task
from bocoel.corpora import Boundary

LOGGER = structlog.get_logger()


class RandomOptimizer(Optimizer):
    def __init__(
        self, query_eval: QueryEvaluator, boundary: Boundary, *, samples: int
    ) -> None:
        LOGGER.info("Instantiating RandomOptimizer", samples=samples)

        self._query_eval = query_eval
        self._boundary = boundary
        self._samples = samples

    @property
    def task(self) -> Task:
        return Task.EXPLORE

    @property
    def terminate(self) -> bool:
        return True

    def step(self) -> Mapping[int, float]:
        lower = self._boundary.lower
        upper = self._boundary.upper

        samples = random.random([self._samples, len(self._boundary)])
        samples *= upper - lower
        samples += lower

        LOGGER.debug("Generated samples", samples=samples, minimum=lower, maximum=upper)

        return self._query_eval(samples)
