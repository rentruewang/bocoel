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
    assert np.isclose(result.distances, 1, atol=1e-5), {
        "results": result,
        "embeddings": embeddings,
    }
    assert np.allclose(result.vectors, query, atol=1e-5), {
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
    index_fix = index(index_string, device)
    embeddings = index_fix.embeddings

    e0 = [embeddings[0]]
    query = [embeddings[0] + embeddings[1] / 2]
    normalized = utils.normalize(query)

    assert normalized.ndim == 2, normalized.shape

    result = index_fix.search(normalized)
    assert np.allclose(result.vectors, e0, atol=1e-5), {
        "results": result,
        "embeddings": embeddings,
    }
    assert result.indices == 0, {
        "results": result,
        "embeddings": embeddings,
    }
