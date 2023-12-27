import pytest
from pandas import DataFrame

from bocoel import DataFrameStorage


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
            {
                "question": "a vastly different question",
                "answer": "a vastly different answer",
            },
        ]
    )


@pytest.fixture
def dataframe_fix() -> DataFrame:
    return df()


def test_dataframe_storage(dataframe_fix: DataFrame) -> None:
    dfs = DataFrameStorage(dataframe_fix)
    assert set(dfs.keys()) == {"question", "answer"}
    assert dfs[0] == dataframe_fix.iloc[0].to_dict()
    assert dfs.get("answer") == dataframe_fix["answer"].to_list()
