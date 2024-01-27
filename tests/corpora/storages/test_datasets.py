from . import factories


def test_datasets_storage() -> None:
    df = factories.df()
    dfs = factories.datasets_storage()
    assert set(dfs.keys()) == {"question", "answer"}
    assert dfs[0] == df.iloc[0].to_dict()
