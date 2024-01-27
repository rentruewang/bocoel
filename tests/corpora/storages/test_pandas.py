from . import factories


def test_dataframe_storage() -> None:
    df = factories.df()
    dfs = factories.df_storage()
    assert set(dfs.keys()) == {"question", "answer"}
    assert dfs[0] == df.iloc[0].to_dict()
