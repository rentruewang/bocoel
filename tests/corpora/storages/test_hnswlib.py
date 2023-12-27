from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from bocoel import HnswlibIndex, Index
from bocoel.corpora.indices import utils as idx_utils


@pytest.fixture
def embeddings() -> NDArray:
    return np.eye(5)


def init_hnswlib_index(embeddings: NDArray) -> Index:
    return HnswlibIndex(embeddings=embeddings, dist="l2")


def test_normalize(embeddings: NDArray) -> None:
    scaled = embeddings * np.array([1, 2, 3, 4, 5])[None, :]
    normalized = idx_utils.normalize(scaled)
    assert np.allclose(normalized, embeddings), {
        "scaled": scaled,
        "normalized": normalized,
        "embeddings": embeddings,
    }


def test_init_hnswlib_index(embeddings: NDArray) -> None:
    index = init_hnswlib_index(embeddings)
    assert index.dims == embeddings.shape[1]
    assert isinstance(index, Index)


def test_hnswlib_index_search_match(embeddings: NDArray) -> None:
    index = init_hnswlib_index(embeddings)

    query = embeddings[0]
    query = idx_utils.normalize(query)

    result = index.search(query)
    assert np.isclose(result.distances, 0), {
        "results": result,
        "embeddings": embeddings,
    }
    assert np.allclose(result.vectors, query), {
        "results": result,
        "embeddings": embeddings,
    }
    assert result.indices == 0, {
        "results": result,
        "embeddings": embeddings,
    }


def test_hnswlib_index_search_mismatch(embeddings: NDArray) -> None:
    index = init_hnswlib_index(embeddings)

    e0 = embeddings[0]
    query = embeddings[0] + embeddings[1] / 2
    query = idx_utils.normalize(query)

    result = index.search(query)
    assert np.allclose(result.vectors, e0), {
        "results": result,
        "embeddings": embeddings,
    }
    assert result.indices == 0, {
        "results": result,
        "embeddings": embeddings,
    }
