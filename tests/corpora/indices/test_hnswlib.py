import numpy as np
import pytest
from numpy.typing import NDArray

from bocoel import Index
from bocoel.corpora.indices import utils

from . import factories


@pytest.fixture(scope="module")
def embeddings_fix() -> NDArray:
    return factories.emb()


@pytest.fixture
def index_fix(embeddings_fix: NDArray) -> Index:
    return factories.hnsw_index(embeddings_fix)


def test_normalize(embeddings_fix: NDArray) -> None:
    scaled = embeddings_fix * np.array([1, 2, 3, 4, 5])[None, :]
    normalized = utils.normalize(scaled)
    assert np.allclose(normalized, embeddings_fix), {
        "scaled": scaled,
        "normalized": normalized,
        "embeddings": embeddings_fix,
    }


def test_init_hnswlib(index_fix: Index, embeddings_fix: NDArray) -> None:
    assert index_fix.dims == embeddings_fix.shape[1]


def test_hnswlib_search_match(index_fix: Index, embeddings_fix: NDArray) -> None:
    query = [embeddings_fix[0]]
    normalized = utils.normalize(query)

    assert normalized.ndim == 2, normalized.shape

    result = index_fix.search(normalized)
    # See https://github.com/nmslib/hnswlib#supported-distances
    assert np.isclose(result.distances, 1 - 1), {
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


def test_hnswlib_search_mismatch(index_fix: Index, embeddings_fix: NDArray) -> None:
    e0 = [embeddings_fix[0]]
    query = [embeddings_fix[0] + embeddings_fix[1] / 2]
    normalized = utils.normalize(query)

    assert normalized.ndim == 2, normalized.shape

    result = index_fix.search(normalized)
    assert np.allclose(result.vectors, e0), {
        "results": result,
        "embeddings": embeddings_fix,
    }
    assert result.indices == 0, {
        "results": result,
        "embeddings": embeddings_fix,
    }
