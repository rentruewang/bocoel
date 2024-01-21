import abc
from abc import ABCMeta
from collections.abc import Mapping
from typing import Any, Protocol

from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.core.optim.evals import QueryEvaluator
from bocoel.core.optim.interfaces import Optimizer, Task
from bocoel.corpora import Boundary


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

    def __init__(self, query_eval: QueryEvaluator, boundary: Boundary) -> None:
        self._query_eval = query_eval
        self._boundary = boundary

    @property
    def task(self) -> Task:
        # Kmeans must be an exploration task.
        return Task.EXPLORE

    @property
    def terminate(self) -> bool:
        return True

    def step(self) -> Mapping[int, float]:
        centers = self._model.cluster_centers_

        return self._query_eval(centers)

    def render(self, **kwargs: Any) -> None:
        raise NotImplementedError
