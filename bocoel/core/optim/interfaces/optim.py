import abc
from collections.abc import Mapping
from typing import Any, Protocol

from .tasks import Task


class Optimizer(Protocol):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Included s.t. constructors of Index can be used.
        ...

    @property
    @abc.abstractmethod
    def task(self) -> Task:
        ...

    @abc.abstractmethod
    def step(self) -> Mapping[int, float]:
        """
        Performs a few steps of optimization.
        Raises StopIteration if done.

        Returns
        -------

        The state change of the optimizer during the step.
        """

        ...

    @abc.abstractmethod
    def render(self, **kwargs: Any) -> None:
        """
        Renders the optimizer's state.
        Parameters are dependent on the underlying configurations.
        """

        ...
