from collections.abc import Mapping

import numpy as np
from cma import CMAEvolutionStrategy

from bocoel.core.optim.evals import QueryEvaluator
from bocoel.core.optim.interfaces import Optimizer
from bocoel.core.tasks import Task
from bocoel.corpora import Boundary


class PyCMAOptimizer(Optimizer):
    """
    TODO: Documentation.

    CMA-ES
    """

    def __init__(
        self,
        query_eval: QueryEvaluator,
        boundary: Boundary,
        *,
        dims: int,
        samples: int,
        minimize: bool = True,
    ) -> None:
        self._query_eval = query_eval

        self._es = CMAEvolutionStrategy(dims * [0], 0.5)
        self._samples = samples
        self._minimize = minimize
        self._boundary = boundary

    @property
    def task(self) -> Task:
        return Task.MINIMIZE if self._minimize else Task.MAXIMIZE

    def step(self) -> Mapping[int, float]:
        if self._es.stop():
            raise StopIteration

        solutions = np.array(self._es.ask(self._samples))

        result = self._query_eval(solutions)

        # This works because result keeps the ordering.
        evaluation = np.array(list(result.values()))

        if not self._minimize:
            evaluation = -evaluation

        self._es.tell(solutions, evaluation)

        return result
