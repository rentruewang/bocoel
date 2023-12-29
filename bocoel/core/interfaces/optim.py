import abc
from typing import Protocol

from bocoel.corpora import Corpus
from bocoel.models import Evaluator, LanguageModel

from .states import State


class Optimizer(Protocol):
    @abc.abstractmethod
    def step(self, corpus: Corpus, lm: LanguageModel, evaluator: Evaluator) -> State:
        ...
