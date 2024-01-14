from collections.abc import Callable, Sequence
from typing import Any

from cma import CMAEvolutionStrategy
from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.core.optim import utils as optim_utils
from bocoel.core.optim.interfaces import Optimizer, State, Task
from bocoel.corpora import Index, SearchResult


class PyCMAOptimizer(Optimizer):
    """
    The sklearn optimizer that uses clustering algorithms.
    See the following webpage for options
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """

    def __init__(
        self,
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
        *,
        samples: int,
    ) -> None:
        self._index = index
        self._evaluate_fn = evaluate_fn

        self._es = CMAEvolutionStrategy(index.dims * [0], 0.5)
        self._samples = samples

    @property
    def task(self) -> Task:
        return Task.MINIMIZE

    @property
    def terminate(self) -> bool:
        return self._es.stop()

    def step(self) -> Sequence[State]:
        solutions = self._es.ask(self._samples)

        result = optim_utils.evaluate_index(
            query=solutions, index=self._index, evaluate_fn=self._evaluate_fn
        )

        self._es.tell(solutions, [r.evaluation for r in result])

        return result

    @classmethod
    def from_index(
        cls,
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
        **kwargs: Any,
    ) -> Self:
        return cls(index=index, evaluate_fn=evaluate_fn, **kwargs)
