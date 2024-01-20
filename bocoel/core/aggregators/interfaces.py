import abc
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from bocoel.core.evals import State
from bocoel.corpora import Corpus
from bocoel.models import Adaptor, LanguageModel


class Agg(Protocol):
    @abc.abstractmethod
    def agg(
        self,
        *,
        corpus: Corpus,
        adaptor: Adaptor,
        lm: LanguageModel,
        states: Sequence[State],
    ) -> Mapping[str, Any]:
        ...
