import abc
from typing import Protocol

from bocoel.corpora import Corpus
from bocoel.models import Evaluator, LanguageModel

from .optim import Optimizer
from .states import State


class Core(Protocol):
    """
    Core is the core of the library.
    It is responsible for finding the best language model, given a corpus.
    It does so by using bayesian optimization to explore the space of different queries.
    Which allows it to find the best query to evaluate the language model on,
    with relatively few evaluations, making the process fast.
    """

    corpus: Corpus
    lm: LanguageModel
    evaluator: Evaluator
    optimizer: Optimizer

    def step(self) -> State:
        return self.optimizer.step(
            corpus=self.corpus, lm=self.lm, evaluator=self.evaluator
        )
