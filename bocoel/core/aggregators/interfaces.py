import abc
from collections.abc import Mapping
from typing import Any, Protocol

from bocoel.corpora import Corpus
from bocoel.models import Adaptor, LanguageModel


class Agg(Protocol):
    @abc.abstractmethod
    def agg(
        self, *, corpus: Corpus, adaptor: Adaptor, lm: LanguageModel
    ) -> Mapping[str, Any]:
        ...
