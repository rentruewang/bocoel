from numpy.typing import NDArray

from bocoel import Distance, HnswlibSearcher, Searcher, WhiteningSearcher

from . import test_hnswlib


def whiten(embeddings: NDArray) -> Searcher:
    return WhiteningSearcher(
        embeddings=embeddings,
        distance=Distance.INNER_PRODUCT,
        remains=3,
        backend=HnswlibSearcher,
        threads=-1,
    )


def test_init_whiten() -> None:
    embeddings = test_hnswlib.emb()
    idx = whiten(embeddings)
    assert idx.dims == 3
