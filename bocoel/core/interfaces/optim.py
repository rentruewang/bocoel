import abc
from typing import Protocol

from bocoel.corpora import Corpus
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
    def step(self, corpus: Corpus, lm: LanguageModel, evaluator: Evaluator) -> State:
        ...
