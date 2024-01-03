from typing import Any

from numpy.typing import NDArray

from bocoel import Distance, HnswlibIndex, Index, WhiteningIndex

from . import test_hnswlib


def whiten_kwargs() -> dict[str, Any]:
    return {
        "distance": Distance.INNER_PRODUCT,
        "remains": 3,
        "backend": HnswlibIndex,
        "threads": -1,
    }


def whiten(embeddings: NDArray) -> Index:
    return WhiteningIndex(embeddings=embeddings, **whiten_kwargs())


def test_init_whiten() -> None:
    embeddings = test_hnswlib.emb()
    idx = whiten(embeddings)
    assert idx.dims == 3
