from . import factories


def test_init_polar() -> None:
    embeddings = factories.emb()
    idx = factories.polar_index(embeddings)
    assert idx.dims == 3 - 1
