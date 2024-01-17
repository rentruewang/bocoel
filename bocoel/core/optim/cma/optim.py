from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from cma import CMAEvolutionStrategy
from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.core.optim import utils as optim_utils
from bocoel.core.optim.interfaces import Optimizer, State, Task
from bocoel.corpora import Index, SearchResult


class PyCMAOptimizer(Optimizer):
    """
    TODO: Documentation.

    CMA-ES
    """

    def __init__(
        self,
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
        *,
        samples: int,
        minimize: bool = True,
    ) -> None:
        self._index = index
        self._evaluate_fn = evaluate_fn

        self._es = CMAEvolutionStrategy(index.dims * [0], 0.5)
        self._samples = samples
        self._minimize = minimize

    @property
    def task(self) -> Task:
        return Task.MINIMIZE if self._minimize else Task.MAXIMIZE

    @property
    def terminate(self) -> bool:
        return self._es.stop()

    def render(self, **kwargs: Any) -> None:
        raise NotImplementedError

    def step(self) -> Sequence[State]:
        solutions = self._es.ask(self._samples)

        result = optim_utils.evaluate_index(
            query=solutions, index=self._index, evaluate_fn=self._evaluate_fn
        )

        evaluation = np.array([r.evaluation for r in result])

        if not self._minimize:
            evaluation = -evaluation

        self._es.tell(solutions, evaluation)

        return result

    @classmethod
    def from_index(
        cls,
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
        **kwargs: Any,
    ) -> Self:
        return cls(index=index, evaluate_fn=evaluate_fn, **kwargs)
