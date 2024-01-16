import abc
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from bocoel.core.optim import State
from bocoel.corpora import Corpus
from bocoel.models import Evaluator, LanguageModel


class Agg(Protocol):
    @abc.abstractmethod
    def agg(
        self,
        *,
        corpus: Corpus,
        evaluator: Evaluator,
        lm: LanguageModel,
        states: Sequence[State],
    ) -> Mapping[str, Any]:
        ...
