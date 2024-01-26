import os
from collections.abc import Sequence
from multiprocessing.pool import ThreadPool

import numpy as np
from numpy.typing import NDArray

from bocoel.corpora.embedders.interfaces import Embedder


class EnsembleEmbedder(Embedder):
    def __init__(self, embedders: Sequence[Embedder]) -> None:
        self._embedders = embedders
        cpus = os.cpu_count()
        assert cpus is not None
        self._cpus = cpus

    @property
    def dims(self) -> int:
        return sum(emb.dims for emb in self._embedders)

    def _encode(self, texts: Sequence[str], /) -> NDArray:
        # Performs mapping in parallel.
        # This only works because pytorch releases GIL during execution.
        with ThreadPool(processes=min(self._cpus, len(self._embedders))) as pool:
            results = pool.starmap(
                _encode_text, [{"emb": emb, "texts": texts} for emb in self._embedders]
            )
        return np.concatenate(results, axis=-1)


def _encode_text(emb: Embedder, texts: Sequence[str]) -> NDArray:
    return emb.encode(texts)
