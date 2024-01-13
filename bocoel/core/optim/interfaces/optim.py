import abc
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.corpora import Corpus, Index, SearchResult
from bocoel.models import Evaluator, LanguageModel

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
    def step(self) -> Sequence[State]:
        """
        Performs a few steps of optimization.

        Returns
        -------

        The state change of the optimizer during the step.
        """

        ...

    @classmethod
    @abc.abstractmethod
    def from_index(
        cls,
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
        **kwargs: Any,
    ) -> Self:
        ...

    @classmethod
    def evaluate_corpus(
        cls, corpus: Corpus, lm: LanguageModel, evaluator: Evaluator, **kwargs: Any
    ) -> Self:
        def evaluate_fn(sr: SearchResult) -> Sequence[float] | NDArray:
            return evaluator.on_corpus(
                corpus=corpus, lm=lm, indices=sr.indices.squeeze(-1)
            ).mean(axis=-1)

        return cls.from_index(index=corpus.index, evaluate_fn=evaluate_fn, **kwargs)
