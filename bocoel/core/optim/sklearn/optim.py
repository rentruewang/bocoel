import abc
from abc import ABCMeta
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.core.optim import utils as optim_utils
from bocoel.core.optim.interfaces import Optimizer, State, Task
from bocoel.corpora import Index, SearchResult


class ScikitLearnCluster(Protocol):
    cluster_centers_: NDArray

    @abc.abstractmethod
    def fit(self, X: Any) -> None:
        ...


class ScikitLearnOptimizer(Optimizer, metaclass=ABCMeta):
    """
    The sklearn optimizer that uses clustering algorithms.
    See the following webpage for options
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """

    _model: ScikitLearnCluster

    def __init__(
        self,
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
    ) -> None:
        self._index = index
        self._evaluate_fn = evaluate_fn

    @property
    def task(self) -> Task:
        # Kmeans must be an exploration task.
        return Task.EXPLORE

    @property
    def terminate(self) -> bool:
        return True

    def step(self) -> Sequence[State]:
        centers = self._model.cluster_centers_

        return optim_utils.evaluate_index(
            query=centers, index=self._index, evaluate_fn=self._evaluate_fn, k=1
        )

    def render(self, **kwargs: Any) -> None:
        raise NotImplementedError

    @classmethod
    def from_index(
        cls,
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
        **kwargs: Any,
    ) -> Self:
        return cls(index=index, evaluate_fn=evaluate_fn, **kwargs)
