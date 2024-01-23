from numpy import random

from . import factories


def test_init_polar() -> None:
    embeddings = factories.emb()
    idx = factories.polar_index(embeddings)
    assert idx.dims == 3 - 1


def test_polar_bounds() -> None:
    NUM_QUERIES = 1000
    K = 3
    embeddings = factories.emb()
    idx = factories.polar_index(embeddings)
    lower = idx.lower
    upper = idx.upper
    rand = random.random(size=[NUM_QUERIES, idx.dims])
    queries = rand * (upper - lower) + lower
    idx.search(queries, k=K)
