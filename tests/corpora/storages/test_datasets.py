from . import factories


def test_datasets_storage() -> None:
    dfs = factories.datasets_storage()
    assert set(dfs.keys()) == {"train", "validation", "test"}
