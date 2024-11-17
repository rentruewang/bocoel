# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import random
from collections.abc import Mapping

import structlog

from bocoel.core.optim.interfaces import IndexEvaluator, Optimizer
from bocoel.core.optim.interfaces.utils import BatchedGenerator
from bocoel.core.tasks import Task
from bocoel.corpora import Index

LOGGER = structlog.get_logger()


class RandomOptimizer(Optimizer):
    """
    The random optimizer that uses random search.
    """

    def __init__(
        self,
        index_eval: IndexEvaluator,
        index: Index,
        *,
        samples: int,
        batch_size: int,
    ) -> None:
        """
        Parameters:
            index_eval: The evaluator to use for the storage.
            index: The index to use for the query.
            samples: The number of samples to use for the optimization.
            batch_size: The number of samples to evaluate at once.
        """

        LOGGER.info("Instantiating RandomOptimizer", total=len(index), samples=samples)

        self._index_eval = index_eval
        self._index = index
        self._generator = iter(BatchedGenerator(self._gen_random(samples), batch_size))

    @property
    def task(self) -> Task:
        return Task.EXPLORE

    @property
    def terminate(self) -> bool:
        return True

    def step(self) -> Mapping[int, float]:
        samples = next(self._generator)
        results = self._index_eval(samples)
        return {s: r for s, r in zip(samples, results)}

    def _gen_random(self, samples: int, /) -> list[int]:
        full = list(range(len(self._index)))
        return random.sample(full, k=samples)
