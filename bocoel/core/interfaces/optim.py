import abc
from collections.abc import Callable
from typing import Any, Protocol

from typing_extensions import Self

from bocoel.corpora import Corpus, Searcher, SearchResult
from bocoel.models import Evaluator

from .states import State


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
    def step(self) -> State:
        """
        Performs a single step of optimization.

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

    @classmethod
    @abc.abstractmethod
    def from_searcher(
        cls,
        searcher: Searcher,
        evaluate_fn: Callable[[SearchResult], float],
        **kwargs: Any
    ) -> Self:
        ...

    @classmethod
    def evaluate_corpus(
        cls, corpus: Corpus, evaluator: Evaluator, **kwargs: Any
    ) -> Self:
        # Import here because implementation depends on interface,
        # and importing at the top-level will cause circular imports.
        from bocoel.core.optim import utils

        return cls.from_searcher(
            searcher=corpus.searcher,
            evaluate_fn=utils.evaluate_corpus_fn(corpus=corpus, evaluator=evaluator),
            **kwargs
        )
