import abc
from collections.abc import Mapping
from typing import Any, Protocol

from bocoel import common
from bocoel.core.tasks import Task
from bocoel.corpora import Boundary

from .evals import QueryEvaluator


class Optimizer(Protocol):
    def __init__(
        self, query_eval: QueryEvaluator, boundary: Boundary, **kwargs: Any
    ) -> None:
        # Included s.t. constructors of Index can be used.
        ...

    def __repr__(self) -> str:
        name = common.remove_base_suffix(self, Optimizer)
        return f"{name}()"

    @property
    @abc.abstractmethod
    def task(self) -> Task: ...

    @abc.abstractmethod
    def step(self) -> Mapping[int, float]:
        """
        Perform a single step of optimization.
        This is a shortcut into the optimization process.
        For methods that evaluate the entire search at once,
        this method would output the slices of the entire search.

        Returns:
            A mapping of step indices to the corresponding scores.

        Raises:
            StopIteration: If the optimization is complete.
        """

        ...
