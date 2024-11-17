# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from collections.abc import Mapping

import numpy as np
from cma import CMAEvolutionStrategy

from bocoel.core.optim.interfaces import IndexEvaluator, Optimizer
from bocoel.core.tasks import Task
from bocoel.corpora import Index


class PyCMAOptimizer(Optimizer):
    """
    Todo: Documentation.

    CMA-ES
    """

    def __init__(
        self,
        index_eval: IndexEvaluator,
        index: Index,
        *,
        dims: int,
        samples: int,
        minimize: bool = True,
    ) -> None:
        self._es = CMAEvolutionStrategy(dims * [0], 0.5)
        self._samples = samples
        self._minimize = minimize

        self._index_eval = index_eval
        self._index = index

    @property
    def task(self) -> Task:
        return Task.MINIMIZE if self._minimize else Task.MAXIMIZE

    def step(self) -> Mapping[int, float]:
        if self._es.stop():
            raise StopIteration

        solutions = np.array(self._es.ask(self._samples))

        indices = self._index.search(query=solutions).indices[..., 0]
        results = self._index_eval(indices)
        returns = {i: r for i, r in zip(indices, results)}

        evaluations = np.array(results) if self._minimize else -np.array(results)

        self._es.tell(solutions, evaluations)
        return returns
