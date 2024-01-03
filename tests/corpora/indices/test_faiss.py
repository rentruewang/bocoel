import numpy as np
import pytest
from numpy.typing import NDArray

from bocoel import Distance, FaissIndex
from bocoel.corpora.indices import utils
from tests import utils as test_utils

from . import test_hnswlib


def index_factory() -> list[str]:
    return ["Flat", "HNSW32"]


@pytest.fixture
def embeddings_fix() -> NDArray:
    return test_hnswlib.emb()


def index(index_string: str, device: str) -> FaissIndex:
    embeddings = test_hnswlib.emb()

    return FaissIndex(
        embeddings=embeddings,
        distance=Distance.INNER_PRODUCT,
        index_string=index_string,
        cuda=device == "cuda",
    )


def test_normalize(embeddings_fix: NDArray) -> None:
    scaled = embeddings_fix * np.array([1, 2, 3, 4, 5])[None, :]
    normalized = utils.normalize(scaled)
    assert np.allclose(normalized, embeddings_fix), {
        "scaled": scaled,
        "normalized": normalized,
        "embeddings": embeddings_fix,
    }


@pytest.mark.parametrize("index_string", index_factory())
@pytest.mark.parametrize("device", test_utils.faiss_devices())
def test_init_faiss(index_string: str, device: str, embeddings_fix: NDArray) -> None:
    search = index(index_string, device)
    assert search.dims == embeddings_fix.shape[1]


@pytest.mark.parametrize("index_string", index_factory())
@pytest.mark.parametrize("device", test_utils.faiss_devices())
def test_faiss_search_match(
    index_string: str, device: str, embeddings_fix: NDArray
) -> None:
    idx = index(index_string, device)

    query = embeddings_fix[0]
    query = utils.normalize(query)

    result = idx.search(query)
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


@pytest.mark.parametrize("index_string", index_factory())
@pytest.mark.parametrize("device", test_utils.faiss_devices())
def test_faiss_search_mismatch(
    index_string: str, device: str, embeddings_fix: NDArray
) -> None:
    index_fix = index(index_string, device)

    e0 = embeddings_fix[0]
    query = embeddings_fix[0] + embeddings_fix[1] / 2
    query = utils.normalize(query)

    result = index_fix.search(query)
    assert np.allclose(result.vectors, e0), {
        "results": result,
        "embeddings": embeddings_fix,
    }
    assert result.indices == 0, {
        "results": result,
        "embeddings": embeddings_fix,
    }
