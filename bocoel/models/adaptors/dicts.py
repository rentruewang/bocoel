from collections.abc import Mapping, Sequence
from typing import Any

from numpy.typing import NDArray

from bocoel.models.adaptors.interfaces import Adaptor, AdaptorBundle
from bocoel.models.lms import LanguageModel


class AdaptorMapping(AdaptorBundle):
    def __init__(self, adaptors: Mapping[str, Adaptor]) -> None:
        self._adaptors = adaptors

    def evaluate(
        self, data: Mapping[str, Sequence[Any]], lm: LanguageModel
    ) -> Mapping[str, Sequence[float] | NDArray]:
        return {name: ev.evaluate(data, lm) for name, ev in self._adaptors.items()}
