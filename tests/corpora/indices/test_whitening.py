from . import factories


def test_init_whiten() -> None:
    embeddings = factories.emb()
    idx = factories.whiten_index(embeddings)
    assert idx.dims == 3
