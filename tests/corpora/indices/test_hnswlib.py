import numpy as np
import pytest
from numpy.typing import NDArray

from bocoel import Distance, HnswlibSearcher, Searcher
from bocoel.corpora.searchers import utils as idx_utils


def emb() -> NDArray:
    return np.eye(5)


@pytest.fixture
def embeddings_fix() -> NDArray:
    return emb()


def index(embeddings: NDArray) -> Searcher:
    return HnswlibSearcher(embeddings=embeddings, distance=Distance.INNER_PRODUCT)


@pytest.fixture
def index_fix(embeddings_fix: NDArray) -> Searcher:
    return index(embeddings_fix)


def test_normalize(embeddings_fix: NDArray) -> None:
    scaled = embeddings_fix * np.array([1, 2, 3, 4, 5])[None, :]
    normalized = idx_utils.normalize(scaled)
    assert np.allclose(normalized, embeddings_fix), {
        "scaled": scaled,
        "normalized": normalized,
        "embeddings": embeddings_fix,
    }


def test_init_hnswlib_index(index_fix: Searcher, embeddings_fix: NDArray) -> None:
    assert index_fix.dims == embeddings_fix.shape[1]


def test_hnswlib_index_search_match(
    index_fix: Searcher, embeddings_fix: NDArray
) -> None:
    query = embeddings_fix[0]
    query = idx_utils.normalize(query)

    result = index_fix.search(query)
    # See https://github.com/nmslib/hnswlib#supported-distances
    assert np.isclose(result.scores, 1 - 1), {
        "results": result,
        "embeddings": embeddings_fix,
    }
    assert np.allclose(result.vectors, query), {
        "results": result,
        "embeddings": embeddings_fix,
    }
    assert result.indices == 0, {
        "results": result,
        "embeddings": embeddings_fix,
    }


def test_hnswlib_index_search_mismatch(
    index_fix: Searcher, embeddings_fix: NDArray
) -> None:
    e0 = embeddings_fix[0]
    query = embeddings_fix[0] + embeddings_fix[1] / 2
    query = idx_utils.normalize(query)

    result = index_fix.search(query)
    assert np.allclose(result.vectors, e0), {
        "results": result,
        "embeddings": embeddings_fix,
    }
    assert result.indices == 0, {
        "results": result,
        "embeddings": embeddings_fix,
    }
