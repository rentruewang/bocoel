import numpy as np
import pytest

from bocoel import Distance, FaissIndex
from bocoel.corpora.indices import utils
from tests import utils as test_utils

from . import factories


def index(index_string: str, device: str) -> FaissIndex:
    embeddings = factories.emb()

    return FaissIndex(
        embeddings=embeddings,
        distance=Distance.INNER_PRODUCT,
        index_string=index_string,
        cuda=device == "cuda",
    )


def test_normalize() -> None:
    embeddings = factories.emb()
    scaled = embeddings * np.array([1, 2, 3, 4, 5])[None, :]
    normalized = utils.normalize(scaled)
    assert np.allclose(normalized, embeddings), {
        "scaled": scaled,
        "normalized": normalized,
        "embeddings": embeddings,
    }


@pytest.mark.parametrize("index_string", factories.index_factory())
@pytest.mark.parametrize("device", test_utils.faiss_devices())
def test_init_faiss(index_string: str, device: str) -> None:
    embeddings = factories.emb()
    search = index(index_string, device)
    assert search.dims == embeddings.shape[1]


@pytest.mark.parametrize("index_string", factories.index_factory())
@pytest.mark.parametrize("device", test_utils.faiss_devices())
def test_faiss_search_match(index_string: str, device: str) -> None:
    embeddings = factories.emb()
    idx = index(index_string, device)

    query = [embeddings[0]]
    normalized = utils.normalize(query)

    assert normalized.ndim == 2, normalized.shape

    result = idx.search(normalized)
    assert np.isclose(result.distances, 1), {
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


@pytest.mark.parametrize("index_string", factories.index_factory())
@pytest.mark.parametrize("device", test_utils.faiss_devices())
def test_faiss_search_mismatch(index_string: str, device: str) -> None:
    embeddings = factories.emb()
    index_fix = index(index_string, device)

    e0 = [embeddings[0]]
    query = [embeddings[0] + embeddings[1] / 2]
    normalized = utils.normalize(query)

    assert normalized.ndim == 2, normalized.shape

    result = index_fix.search(normalized)
    assert np.allclose(result.vectors, e0), {
        "results": result,
        "embeddings": embeddings,
    }
    assert result.indices == 0, {
        "results": result,
        "embeddings": embeddings,
    }
