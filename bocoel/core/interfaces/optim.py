import abc
from typing import Any, ParamSpec, Protocol

from bocoel.corpora import Corpus
from bocoel.models import Evaluator, LanguageModel

from .states import State

P = ParamSpec("P")


class Optimizer(Protocol):
    @property
    @abc.abstractmethod
    def terminate(self) -> bool:
        """
        Terminate decides if the optimization loop should terminate early.
        If terminate = False, the optimization loop will continue to the given iteration.
        """

        ...

    @abc.abstractmethod
    def step(self, corpus: Corpus, lm: LanguageModel, evaluator: Evaluator) -> State:
        """
        Performs a single step of optimization.

        Parameters
        ----------

        `corpus: Corpus`

        `lm: LanguageModel`

        `evaluator: Evaluator`

        Returns
        -------

        The state change of the optimizer during the step.
        """

        ...

    @abc.abstractmethod
    def render(self, kind: str, **kwargs: Any) -> None:
        """
        Renders the history of the optimizer for debug use.

        Parameters
        ----------

        `kind: str`
        The type of rendering to perform. Is dependent on optimizer.
        See documentation for each render for more.

        `**kwargs: Any`
        Additional arguments to pass to the rendering function.
        """

        ...
