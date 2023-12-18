import abc
from typing import Protocol

from numpy.typing import NDArray

from bocoel.corpora import Corpus
from bocoel.models import LanguageModel

from .histories import History


# FIXME: Depends on what is the most convenient, might need redesign.
class Experiment(Protocol):
    key: str
    corpus: Corpus
    llm: LanguageModel
    history: History
    done: bool

    def optimize(self):
        index = self.corpus.keys_index[self.key]

        while not self.done:
            target = self.acquisition_function()
            best_idx = index(target).item()
            best_prompt = self.corpus[best_idx]
            text = self.llm.generate(best_prompt[self.key])

    @abc.abstractmethod
    def acquisition_function(self) -> NDArray:
        ...
