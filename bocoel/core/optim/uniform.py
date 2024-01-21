import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray

from bocoel.core.optim.evals import QueryEvaluator
from bocoel.core.optim.interfaces import Optimizer, Task
from bocoel.corpora import Boundary

LOGGER = structlog.get_logger()


class UniformOptimizer(Optimizer):
    def __init__(
        self, query_eval: QueryEvaluator, boundary: Boundary, *, grids: Sequence[int]
    ) -> None:
        LOGGER.info("Instantiating UnfiromOptimizer", grids=grids)

        self._query_eval = query_eval
        self._boundary = boundary
        self._grids = grids

        if len(self._grids) != self._boundary.dims:
            raise ValueError(
                f"Expected {self._boundary.dims} strides, got {self._grids}"
            )

    @property
    def task(self) -> Task:
        return Task.EXPLORE

    @property
    def terminate(self) -> bool:
        return True

    def step(self) -> Mapping[int, float]:
        samples = self._generate_queries()
        return self._query_eval(samples)

    def _generate_queries(self) -> NDArray:
        lower = self._boundary.lower
        upper = self._boundary.upper

        box_size = upper - lower
        step_size = box_size / self._grids

        return np.array(
            [
                step_size * (np.array(combo) + 0.5)
                for combo in itertools.product(*[range(grid) for grid in self._grids])
            ]
        )

    def render(self, **kwargs: Any) -> None:
        raise NotImplementedError
