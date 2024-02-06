import abc
from collections.abc import Mapping
from typing import Any, Protocol

from bocoel.core.tasks import Task
from bocoel.corpora import Boundary

from .evals import QueryEvaluator

_VERSION_KEY = "version"
_OPTIMIZER_KEY = "optimizer"


class Optimizer(Protocol):
    def __init__(
        self, query_eval: QueryEvaluator, boundary: Boundary, **kwargs: Any
    ) -> None:
        # Included s.t. constructors of Index can be used.
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    @property
    @abc.abstractmethod
    def task(self) -> Task: ...

    @abc.abstractmethod
    def step(self) -> Mapping[int, float]:
        """
        Perform a single step of optimization.

        Returns:
            A mapping of step indices to the corresponding scores.

        Raises:
            StopIteration: If the optimization is complete.
        """

        ...
