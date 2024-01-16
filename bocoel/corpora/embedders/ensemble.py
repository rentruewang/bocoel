from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from bocoel.corpora.embedders.interfaces import Embedder


class EnsembleEmbedder(Embedder):
    def __init__(self, embedders: Sequence[Embedder]) -> None:
        self._embedders = embedders

    @property
    def dims(self) -> int:
        return sum(emb.dims for emb in self._embedders)

    def _encode(self, texts: Sequence[str], /) -> NDArray:
        return np.concatenate([emb._encode(texts) for emb in self._embedders], axis=0)
