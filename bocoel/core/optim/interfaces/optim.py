# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
from collections.abc import Mapping
from typing import Any, Protocol

from bocoel import common
from bocoel.core.tasks import Task
from bocoel.corpora import Index

from .evals import IndexEvaluator


class Optimizer(Protocol):
    """
    The protocol for optimizers.
    Optimizers are used for optimizing the search space,
    Find the best exploration sequence for a given task.
    """

    def __init__(self, index_eval: IndexEvaluator, index: Index, **kwargs: Any) -> None:
        """
        Parameters:
            index_eval: The id evaluator. Evalutes the items at the given storage indices.
            index: The index that contains information about the domain.
            **kwargs: The keyword arguments.
        """

        # Included s.t. constructors of Index can be used.
        ...

    def __repr__(self) -> str:
        name = common.remove_base_suffix(self, Optimizer)
        return f"{name}()"

    @property
    @abc.abstractmethod
    def task(self) -> Task:
        """
        The task to use for the optimization.

        Returns:
            One of `Task.EXPLORE` or `Task.MINIMIZE` or `Task.MAXIMIZE`.
        """

        ...

    @abc.abstractmethod
    def step(self) -> Mapping[int, float]:
        """
        Perform a single step of optimization.
        This is a shortcut into the optimization process.
        For methods that evaluate the entire search at once,
        this method would output the slices of the entire search.

        Returns:
            A mapping of storage indices to the corresponding scores.

        Raises:
            StopIteration: If the optimization is complete.
        """

        ...
