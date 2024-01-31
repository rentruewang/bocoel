from datasets import Dataset
from pandas import DataFrame

from bocoel import DatasetsStorage, PandasStorage, Storage
from tests import utils


@utils.cache
def df() -> DataFrame:
    return DataFrame.from_records(
        [
            {
                "question": "hello is anyone home?",
                "answer": ["no"],
            },
            {
                "question": "what's up?",
                "answer": ["not much"],
            },
            {
                "question": "who is the most handsome man in the world?",
                "answer": ["it's you, renchu wang"],
            },
            {
                "question": "a vastly different question",
                "answer": ["a vastly different answer"],
            },
        ]
    )


@utils.cache
def df_storage() -> Storage:
    return PandasStorage(df())


def dataset() -> Dataset:
    return Dataset.from_pandas(df())


@utils.cache
def datasets_storage() -> Storage:
    return DatasetsStorage(dataset())
