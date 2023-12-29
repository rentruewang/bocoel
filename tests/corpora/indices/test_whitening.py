from numpy.typing import NDArray

from bocoel import Distance, HnswlibIndex, Index, WhiteningIndex

from . import test_hnswlib


def whiten_index(embeddings: NDArray) -> Index:
    return WhiteningIndex(
        embeddings=embeddings,
        distance=Distance.INNER_PRODUCT,
        remains=3,
        idx_cls=HnswlibIndex,
        threads=-1,
    )


def test_init_whiten_index() -> None:
    embeddings = test_hnswlib.emb()
    idx = whiten_index(embeddings)
    assert idx.dims == 3
