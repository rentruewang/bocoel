import abc
from abc import ABCMeta
from collections.abc import Mapping
from typing import Any, Protocol

from numpy.typing import NDArray

from bocoel.core.optim.evals import QueryEvaluator
from bocoel.core.optim.interfaces import Optimizer
from bocoel.core.optim.utils import BatchedGenerator
from bocoel.core.tasks import Task
from bocoel.corpora import Boundary


class ScikitLearnCluster(Protocol):
    """
    The protocol for clustering models in scikit-learn.
    """

    cluster_centers_: NDArray

    @abc.abstractmethod
    def fit(self, X: Any) -> None: ...


class ScikitLearnOptimizer(Optimizer, metaclass=ABCMeta):
    """
    The sklearn optimizer that uses clustering algorithms.
    See the following webpage for options
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """

    def __init__(
        self,
        query_eval: QueryEvaluator,
        boundary: Boundary,
        model: ScikitLearnCluster,
        batch_size: int,
    ) -> None:
        """
        Parameters:
            query_eval: The evaluator to use for the query.
            boundary: The boundary to use for the query.
            model: The model to use for the optimization.
            batch_size: The number of embeddings to evaluate at once.
        """

        self._query_eval = query_eval
        self._boundary = boundary

        self._generator = iter(BatchedGenerator(model.cluster_centers_, batch_size))

    @property
    def task(self) -> Task:
        # Kmeans must be an exploration task.
        return Task.EXPLORE

    def step(self) -> Mapping[int, float]:
        centers = next(self._generator)
        return self._query_eval(centers)
