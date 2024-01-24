import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.core.optim.evals import QueryEvaluator
from bocoel.core.optim.interfaces import Optimizer, Task

LOGGER = structlog.get_logger()


class UniformOptimizer(Optimizer):
    def __init__(
        self, query_eval: QueryEvaluator, bounds: NDArray, *, grids: Sequence[int]
    ) -> None:
        LOGGER.info("Instantiating UnfiromOptimizer", grids=grids)

        self._query_eval = query_eval
        # FIXME: Use better abstractions.
        self._bounds = bounds
        self._grids = grids

        if len(self._grids) != len(self._bounds):
            raise ValueError(f"Expected {len(self._bounds)} strides, got {self._grids}")

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
        lower = self._bounds[:, 0]
        upper = self._bounds[:, 1]

        box_size = upper - lower
        step_size = box_size / self._grids

        return np.array(
            [
                step_size * (np.array(combo) + 0.5)
                for combo in itertools.product(*[range(grid) for grid in self._grids])
            ]
        )

    @classmethod
    def from_stateful_eval(cls, evaluate_fn: QueryEvaluator, /, **kwargs: Any) -> Self:
        return cls(query_eval=evaluate_fn, **kwargs)
