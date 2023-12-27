from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import ujson as json
from pandas import DataFrame
from typing_extensions import Self

from bocoel.corpora.interfaces import Storage


class DataFrameStorage(Storage):
    def __init__(self, df: DataFrame) -> None:
        self._df = df

    def keys(self) -> Collection[str]:
        return self._df.columns

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Mapping[str, str]:
        return self._df.iloc[idx].to_dict()

    def get(self, key: str) -> Sequence[str]:
        return self._df[key].to_list()

    @classmethod
    def from_jsonl_file(cls, path: str | Path) -> Self:
        path = Path(path)

        # TODO: Also support directories.
        assert path.is_file()

        with open(path) as f:
            lines = map(lambda s: s.strip("\n"), f.readlines())

        data = [json.loads(line) for line in lines]
        return cls.from_jsonl(data)

    @classmethod
    def from_jsonl(cls, data: Sequence[Mapping[str, str]]) -> Self:
        df = DataFrame.from_records(data)
        return cls(df)
