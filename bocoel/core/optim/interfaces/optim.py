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

    @property
    @abc.abstractmethod
    def terminate(self) -> bool:
        """
        Terminate decides if the optimization loop should terminate early.
        If terminate = False, the optimization loop will continue to the given iteration.
        """

        ...

    @abc.abstractmethod
    def step(self) -> Mapping[int, float]:
        """
        Performs a few steps of optimization.

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
