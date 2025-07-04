# Copyright (c) BoCoEL Authors - All Rights Reserved

from collections.abc import Mapping

import structlog

from bocoel.corpora import Index
from bocoel.optim.interfaces import IndexEvaluator, Optimizer
from bocoel.optim.interfaces.utils import BatchedGenerator
from bocoel.tasks import Task

__all__ = ["BruteForceOptimizer"]

LOGGER = structlog.get_logger()


class BruteForceOptimizer(Optimizer):
    def __init__(
        self,
        index_eval: IndexEvaluator,
        index: Index,
        *,
        total: int,
        batch_size: int,
    ) -> None:
        LOGGER.info("Instantiating CompleteOptimizer", samples=total)

        self._index_eval = index_eval
        self._index = index
        self._generator = iter(BatchedGenerator(range(total), batch_size))

    @property
    def task(self) -> Task:
        return Task.EXPLORE

    @property
    def terminate(self) -> bool:
        return True

    def step(self) -> Mapping[int, float]:
        indices = next(self._generator)
        results = self._index_eval(indices)
        return {i: r for i, r in zip(indices, results)}
