# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
from abc import ABCMeta
from collections.abc import Mapping
from typing import Any, Protocol

from numpy.typing import NDArray
from sklearn.utils import validation

from bocoel.core.optim.interfaces import IndexEvaluator, Optimizer
from bocoel.core.optim.interfaces.utils import BatchedGenerator
from bocoel.core.tasks import Task
from bocoel.corpora import Index


class ScikitLearnCluster(Protocol):
    """
    The protocol for clustering models in scikit-learn.
    """

    cluster_centers_: NDArray

    @abc.abstractmethod
    def fit(self, X: Any) -> None:
        """
        Fits the model and returns the cluster indices.

        Parameters:
            X: The data to fit.
        """

        ...

    @abc.abstractmethod
    def predict(self, X: Any) -> list[int] | NDArray:
        """
        Fits the model and returns the cluster indices.

        Parameters:
            X: The data to use.

        Returns:
            The cluster indices.
        """

        ...


class ScikitLearnOptimizer(Optimizer, metaclass=ABCMeta):
    """
    The sklearn optimizer that uses clustering algorithms.
    See the following webpage for options
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """

    def __init__(
        self,
        index_eval: IndexEvaluator,
        index: Index,
        embeddings: NDArray,
        model: ScikitLearnCluster,
        batch_size: int,
    ) -> None:
        """
        Parameters:
            index_eval: The evaluator to use for the query.
            index: The index to use for the query.
            model: The model to use for the optimization.
            batch_size: The number of embeddings to evaluate at once.
        """

        self._index_eval = index_eval
        self._index = index

        model.fit(embeddings)
        validation.check_is_fitted(model)
        ids = model.predict(model.cluster_centers_)

        self._generator = iter(BatchedGenerator(ids, batch_size))

    @property
    def task(self) -> Task:
        # Kmeans must be an exploration task.
        return Task.EXPLORE

    def step(self) -> Mapping[int, float]:
        indices = next(self._generator)
        results = self._index_eval(indices)
        return {i: r for i, r in zip(indices, results)}
