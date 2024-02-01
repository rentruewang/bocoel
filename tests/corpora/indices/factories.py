from typing import Any

import numpy as np
from numpy import random
from numpy.typing import NDArray

from bocoel import Distance, HnswlibIndex, Index, PolarIndex, WhiteningIndex
from bocoel.corpora.indices import utils as index_utils
from tests import utils


def index_factory() -> list[str]:
    return ["Flat", "HNSW32"]


@utils.cache
def emb() -> NDArray:
    return index_utils.normalize(random.randn(7, 5) + np.arange(7)[:, None])


def hnsw_index(embeddings: NDArray) -> Index:
    return HnswlibIndex(embeddings=embeddings, distance=Distance.INNER_PRODUCT)


def hnswlib_kwargs() -> dict[str, Any]:
    return {"threads": -1}


def whiten_kwargs() -> dict[str, Any]:
    return {"reduced": 3, "whitening_backend": HnswlibIndex, **hnswlib_kwargs()}


def polar_kwargs() -> dict[str, Any]:
    return {"polar_backend": WhiteningIndex, **whiten_kwargs()}


def whiten_index(embeddings: NDArray, /) -> Index:
    return WhiteningIndex(
        embeddings=embeddings, distance=Distance.INNER_PRODUCT, **whiten_kwargs()
    )


def polar_index(embeddings: NDArray, /) -> Index:
    return PolarIndex(
        embeddings=embeddings, distance=Distance.INNER_PRODUCT, **polar_kwargs()
    )
