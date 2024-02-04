import abc
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from packaging import version
from typing_extensions import Self

from bocoel import common
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
        Performs a few steps of optimization.
        Raises StopIteration if done.

        Returns
        -------

        The state change of the optimizer during the step.
        """

        ...

    def save(self, path: str | Path) -> None:
        """
        Saves the optimizer to a file.

        Parameters
        ----------

        `path: str | Path`
        The path to save the optimizer to.
        """

        with open(path, "wb") as f:
            pickle.dump({_VERSION_KEY: common.version(), _OPTIMIZER_KEY: self}, f)

    def load(self, path: str | Path) -> Self:
        current_version = version.parse(common.version())

        with open(path, "rb") as f:
            loaded = pickle.load(f)

        loaded_version = version.parse(loaded[_VERSION_KEY])

        if (
            False
            or current_version.major != loaded_version.major
            or current_version.minor != loaded_version.minor
        ):
            raise ValueError(f"Version mismatch: {current_version} != {loaded_version}")

        return loaded[_OPTIMIZER_KEY]
