import abc
from collections.abc import Callable
from typing import Any, Protocol

from typing_extensions import Self

from bocoel.corpora import Corpus, Index, SearchResult
from bocoel.models import Score

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

    @classmethod
    @abc.abstractmethod
    def from_index(
        cls, index: Index, evaluate_fn: Callable[[SearchResult], float], **kwargs: Any
    ) -> Self:
        ...

    @classmethod
    def evaluate_corpus(cls, corpus: Corpus, scorer: Score, **kwargs: Any) -> Self:
        # Import here because implementation depends on interface,
        # and importing at the top-level will cause circular imports.
        from bocoel.core.optim import utils

        return cls.from_index(
            index=corpus.index,
            evaluate_fn=utils.evaluate_corpus_fn(corpus=corpus, evaluator=scorer),
            **kwargs
        )
