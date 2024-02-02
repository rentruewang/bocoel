import itertools
from collections.abc import Generator, Mapping, Sequence

import numpy as np
import structlog
from numpy.typing import NDArray

from bocoel.core.optim.evals import QueryEvaluator
from bocoel.core.optim.interfaces import Optimizer
from bocoel.core.optim.utils import BatchedGenerator
from bocoel.core.tasks import Task
from bocoel.corpora import Boundary

LOGGER = structlog.get_logger()


class UniformOptimizer(Optimizer):
    def __init__(
        self,
        query_eval: QueryEvaluator,
        boundary: Boundary,
        *,
        grids: Sequence[int],
        batch_size: int,
    ) -> None:
        LOGGER.info("Instantiating UnfiromOptimizer", grids=grids)

        self._query_eval = query_eval
        self._boundary = boundary

        self._generator = iter(BatchedGenerator(self._gen_locs(grids), batch_size))

        if len(grids) != self._boundary.dims:
            raise ValueError(f"Expected {self._boundary.dims} strides, got {grids}")

    @property
    def task(self) -> Task:
        return Task.EXPLORE

    def step(self) -> Mapping[int, float]:
        locs = next(self._generator)
        return self._query_eval(locs)

    def _gen_locs(self, grids: Sequence[int]) -> Generator[NDArray, None, None]:
        lower = self._boundary.lower
        upper = self._boundary.upper

        box_size = upper - lower
        step_size = box_size / np.array(grids)

        for combo in itertools.product(*[range(grid) for grid in grids]):
            yield step_size * (np.array(combo) + 0.5)
