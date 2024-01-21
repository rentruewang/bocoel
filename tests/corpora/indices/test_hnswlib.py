import numpy as np

from bocoel import Index
from bocoel.corpora.indices import utils

from . import factories


def get_index() -> Index:
    embeddings = factories.emb()

    return factories.hnsw_index(embeddings)


def test_init_hnswlib() -> None:
    embeddings = factories.emb()

    assert get_index().dims == embeddings.shape[1]


def test_hnswlib_search_match() -> None:
    embeddings = factories.emb()

    query = [embeddings[0]]
    normalized = utils.normalize(query)

    assert normalized.ndim == 2, normalized.shape

    result = get_index().search(normalized)
    # See https://github.com/nmslib/hnswlib#supported-distances
    assert np.isclose(result.distances, 1 - 1, atol=1e-5), {
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


def test_hnswlib_search_mismatch() -> None:
    embeddings = factories.emb()

    e0 = [embeddings[0]]
    query = [embeddings[0] + embeddings[1] / 2]
    normalized = utils.normalize(query)

    assert normalized.ndim == 2, normalized.shape

    result = get_index().search(normalized)
    assert np.allclose(result.vectors, e0, atol=1e-5), {
        "results": result,
        "embeddings": embeddings,
    }
    assert result.indices == 0, {
        "results": result,
        "embeddings": embeddings,
    }
