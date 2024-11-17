# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import itertools
from collections.abc import Generator, Mapping, Sequence

import numpy as np
import structlog
from numpy.typing import NDArray

from bocoel.core.optim.interfaces import IndexEvaluator, Optimizer
from bocoel.core.optim.interfaces.utils import BatchedGenerator
from bocoel.core.tasks import Task
from bocoel.corpora import Index

LOGGER = structlog.get_logger()


class UniformOptimizer(Optimizer):
    """
    The uniform optimizer that uses grid-based search.
    """

    def __init__(
        self,
        index_eval: IndexEvaluator,
        index: Index,
        *,
        grids: Sequence[int],
        batch_size: int,
    ) -> None:
        """
        Parameters:
            index_eval: The evaluator to use for the storage.
            index: The index to use for the query.
            grids: The number of grids to use for the optimization.
            batch_size: The number of grids to evaluate at once.
        """

        LOGGER.info("Instantiating UnfiromOptimizer", grids=grids)

        self._index_eval = index_eval
        self._index = index

        self._generator = iter(BatchedGenerator(self._gen_locs(grids), batch_size))

        if len(grids) != self._index.dims:
            raise ValueError(f"Expected {self._index.dims} strides, got {grids}")

    @property
    def task(self) -> Task:
        return Task.EXPLORE

    def step(self) -> Mapping[int, float]:
        locs = next(self._generator)
        indices = self._index.search(query=locs).indices
        results = self._index_eval(indices)[..., 0]
        return {i: r for i, r in zip(indices, results)}

    def _gen_locs(self, grids: Sequence[int]) -> Generator[NDArray, None, None]:
        lower = self._index.lower
        upper = self._index.upper

        box_size = upper - lower
        step_size = box_size / np.array(grids)

        for combo in itertools.product(*[range(grid) for grid in grids]):
            yield step_size * (np.array(combo) + 0.5)
