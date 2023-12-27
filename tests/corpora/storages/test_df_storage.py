import pytest
from pandas import DataFrame

from bocoel import DataFrameStorage


@pytest.fixture
def df() -> DataFrame:
    return DataFrame.from_records(
        [
            {
                "question": "hello is anyone home?",
                "answer": "no",
            },
            {
                "question": "what's up?",
                "answer": "not much",
            },
            {
                "question": "who is the most handsome man in the world?",
                "answer": "it's you, renchu wang",
            },
        ]
    )


def test_dataframe_storage(df: DataFrame) -> None:
    dfs = DataFrameStorage(df)
    assert set(dfs.keys()) == {"question", "answer"}
    assert len(dfs) == len(df) == 3
    assert dfs[0] == df.iloc[0].to_dict()
    assert dfs.get("answer") == df["answer"].to_list()
