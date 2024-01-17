import numpy as np

from bocoel.corpora.indices import utils


def test_normalize() -> None:
    embeddings = np.eye(5)
    scaled = embeddings * np.array([1, 2, 3, 4, 5])[None, :]
    normalized = utils.normalize(scaled)
    assert np.allclose(normalized, embeddings), {
        "scaled": scaled,
        "normalized": normalized,
        "embeddings": embeddings,
    }
