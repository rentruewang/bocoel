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
                "answer_id": 1,
            },
            {
                "question": "what's up?",
                "answer": ["not much"],
                "answer_id": 0,
            },
            {
                "question": "who is the most handsome man in the world?",
                "answer": ["it's you, renchu wang"],
                "answer_id": 0,
            },
            {
                "question": "a vastly different question",
                "answer": ["a vastly different answer"],
                "answer_id": 1,
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
    return DatasetsStorage("SetFit/sst2")
