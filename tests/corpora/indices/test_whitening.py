from numpy.typing import NDArray

from bocoel import Distance, HnswlibSearcher, Searcher, WhiteningSearcher

from . import test_hnswlib


def whiten_index(embeddings: NDArray) -> Searcher:
    return WhiteningSearcher(
        embeddings=embeddings,
        distance=Distance.INNER_PRODUCT,
        remains=3,
        idx_cls=HnswlibSearcher,
        threads=-1,
    )


def test_init_whiten_index() -> None:
    embeddings = test_hnswlib.emb()
    idx = whiten_index(embeddings)
    assert idx.dims == 3
