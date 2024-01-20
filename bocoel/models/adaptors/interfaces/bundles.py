import abc
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from numpy.typing import NDArray

from bocoel.models.lms import LanguageModel


class AdaptorBundle(Protocol):
    @abc.abstractmethod
    def evaluate(
        self, data: Mapping[str, Sequence[Any]], lm: LanguageModel
    ) -> Mapping[str, Sequence[float] | NDArray]:
        ...
