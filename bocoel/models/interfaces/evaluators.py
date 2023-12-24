from __future__ import annotations

import abc
from typing import Any, Mapping, Protocol, Sequence

from bocoel.corpora import Corpus, Storage
from bocoel.models.interfaces import LanguageModel


class Evaluator(Protocol):
    def __call__(self, lm: LanguageModel, corpus: Corpus, /, *keys: str) -> Any:
        self._validate_inputs(*keys)
        return self.eval(lm, corpus.storage, *keys)

    @abc.abstractmethod
    def eval(self, lm: LanguageModel, store: Storage, /, *keys: str) -> Sequence[float]:
        ...

    @abc.abstractmethod
    def _validate_inputs(self, *keys: str) -> None:
        ...
