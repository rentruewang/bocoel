import numpy as np
import pytest
from numpy.typing import NDArray
from pytest import FixtureRequest

from bocoel import Distance, FaissSearcher, Searcher
from bocoel.corpora.searchers import utils as idx_utils

from . import test_hnswlib


def index_factory() -> list[str]:
    return ["Flat", "HNSW32"]


@pytest.fixture
def embeddings_fix() -> NDArray:
    return test_hnswlib.emb()


@pytest.fixture
def index_fix(request: FixtureRequest) -> FaissSearcher:
    embeddings = test_hnswlib.emb()

    return FaissSearcher(
        embeddings=embeddings,
        distance=Distance.INNER_PRODUCT,
        index_string=request.param,
    )


def test_normalize(embeddings_fix: NDArray) -> None:
    scaled = embeddings_fix * np.array([1, 2, 3, 4, 5])[None, :]
    normalized = idx_utils.normalize(scaled)
    assert np.allclose(normalized, embeddings_fix), {
        "scaled": scaled,
        "normalized": normalized,
        "embeddings": embeddings_fix,
    }


@pytest.mark.parametrize("index_fix", index_factory(), indirect=True)
def test_init_faiss_index(index_fix: Searcher, embeddings_fix: NDArray) -> None:
    assert index_fix.dims == embeddings_fix.shape[1]


@pytest.mark.parametrize("index_fix", index_factory(), indirect=True)
def test_faiss_index_search_match(index_fix: Searcher, embeddings_fix: NDArray) -> None:
    query = embeddings_fix[0]
    query = idx_utils.normalize(query)

    result = index_fix.search(query)
    assert np.isclose(result.scores, 1), {
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


@pytest.mark.parametrize("index_fix", index_factory(), indirect=True)
def test_faiss_index_search_mismatch(
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
