from collections.abc import Generator, Mapping

import structlog
from numpy import random
from numpy.typing import NDArray

from bocoel.core.optim.evals import QueryEvaluator
from bocoel.core.optim.interfaces import Optimizer
from bocoel.core.optim.utils import BatchedGenerator
from bocoel.core.tasks import Task
from bocoel.corpora import Boundary

LOGGER = structlog.get_logger()


class RandomOptimizer(Optimizer):
    def __init__(
        self,
        query_eval: QueryEvaluator,
        boundary: Boundary,
        *,
        samples: int,
        batch_size: int,
    ) -> None:
        LOGGER.info("Instantiating RandomOptimizer", samples=samples)

        self._query_eval = query_eval
        self._boundary = boundary
        self._generator = iter(BatchedGenerator(self._gen_random(samples), batch_size))

    @property
    def task(self) -> Task:
        return Task.EXPLORE

    @property
    def terminate(self) -> bool:
        return True

    def step(self) -> Mapping[int, float]:
        samples = next(self._generator)
        return self._query_eval(samples)

    def _gen_random(self, samples: int, /) -> Generator[NDArray, None, None]:
        lower = self._boundary.lower
        upper = self._boundary.upper

        for _ in range(samples):
            point = random.random([len(self._boundary)])
            point *= upper - lower
            point += lower
            yield point
