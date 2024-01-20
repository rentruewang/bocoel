import abc
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.core.optim import evals
from bocoel.core.optim.evals import QueryEvaluator, ResultEvaluator
from bocoel.corpora import Corpus, SearchResultBatch, StatefulIndex
from bocoel.models import Adaptor, LanguageModel

from .tasks import Task


class Optimizer(Protocol):
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

    @classmethod
    @abc.abstractmethod
    def from_stateful_eval(cls, evaluate_fn: QueryEvaluator, /, **kwargs: Any) -> Self:
        ...

    @classmethod
    @abc.abstractmethod
    def from_index(
        cls, index: StatefulIndex, evaluate_fn: ResultEvaluator, **kwargs: Any
    ) -> Self:
        query_eval = evals.query_eval_func(index, evaluate_fn)
        return cls.from_stateful_eval(query_eval, **kwargs)

    @classmethod
    def evaluate_corpus(
        cls, corpus: Corpus, lm: LanguageModel, adaptor: Adaptor, **kwargs: Any
    ) -> Self:
        def evaluate_fn(sr: SearchResultBatch, /) -> Sequence[float] | NDArray:
            evaluated = adaptor.on_corpus(corpus=corpus, lm=lm, indices=sr.indices)
            assert (
                evaluated.ndim == 2
            ), f"Evaluated should have the dimensions [batch, k]. Got {evaluated.shape}"
            return evaluated.mean(axis=-1)

        return cls.from_index(
            index=corpus.index,
            evaluate_fn=evals.stateful_eval_func(evaluate_fn),
            **kwargs,
        )
