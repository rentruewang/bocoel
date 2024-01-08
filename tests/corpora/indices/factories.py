from typing import Any

import numpy as np
from numpy.typing import NDArray

from bocoel import Distance, HnswlibIndex, Index, PolarIndex, WhiteningIndex


def index_factory() -> list[str]:
    return ["Flat", "HNSW32"]


def emb() -> NDArray:
    return np.eye(5)


def hnsw_index(embeddings: NDArray) -> Index:
    return HnswlibIndex(embeddings=embeddings, distance=Distance.INNER_PRODUCT)


def hnswlib_kwargs() -> dict[str, Any]:
    return {
        "threads": -1,
    }


def whiten_kwargs() -> dict[str, Any]:
    return {"remains": 3, "backend": HnswlibIndex, "backend_kwargs": hnswlib_kwargs()}


def polar_kwargs() -> dict[str, Any]:
    return {"backend": WhiteningIndex, "backend_kwargs": whiten_kwargs()}


def whiten_index(embeddings: NDArray) -> Index:
    return WhiteningIndex(
        embeddings=embeddings, distance=Distance.INNER_PRODUCT, **whiten_kwargs()
    )


def polar_index(embeddings: NDArray) -> Index:
    return PolarIndex(
        embeddings=embeddings, distance=Distance.INNER_PRODUCT, **polar_kwargs()
    )
