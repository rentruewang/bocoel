import itertools
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.core.evals import State
from bocoel.core.optim import utils as optim_utils
from bocoel.core.optim.interfaces import Optimizer, Task
from bocoel.corpora import Index, SearchResult

LOGGER = structlog.get_logger()


class UniformOptimizer(Optimizer):
    def __init__(
        self,
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
        *,
        grids: Sequence[int],
    ) -> None:
        LOGGER.info("Instantiating UnfiromOptimizer", grids=grids)

        self._index = index
        self._evaluate_fn = evaluate_fn
        self._grids = grids

        if len(self._grids) != self._index.dims:
            raise ValueError(f"Expected {self._index.dims} strides, got {self._grids}")

    @property
    def task(self) -> Task:
        return Task.EXPLORE

    @property
    def terminate(self) -> bool:
        return True

    def step(self) -> Sequence[State]:
        samples = self._generate_queries()
        return optim_utils.evaluate_index(
            query=samples, index=self._index, evaluate_fn=self._evaluate_fn
        )

    def _generate_queries(self) -> NDArray:
        lower = self._index.lower
        upper = self._index.upper

        box_size = upper - lower
        step_size = box_size / self._grids

        return np.array(
            [
                step_size * (np.array(combo) + 0.5)
                for combo in itertools.product(*[range(grid) for grid in self._grids])
            ]
        )

    @classmethod
    def from_index(
        cls,
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
        **kwargs: Any,
    ) -> Self:
        return cls(index=index, evaluate_fn=evaluate_fn, **kwargs)
