from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from bocoel.corpora.embedders.interfaces import Embedder


class Ensemblembedder(Embedder):
    def __init__(
        self, callables: Sequence[Callable[[Sequence[str]], Sequence[float]]]
    ) -> None:
        self._callables = callables

    @property
    def dims(self) -> int:
        return len(self._callables)

    def _encode(self, text: Sequence[str]) -> NDArray:
        return np.array([call(text) for call in self._callables]).T
