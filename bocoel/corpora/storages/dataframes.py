from __future__ import annotations

import typing
from pathlib import Path
from typing import Container, Mapping, Sequence

import ujson as json
from pandas import DataFrame

from bocoel.corpora.interfaces import Storage


class DataFrameStorage(Storage):
    def __init__(self, data: DataFrame) -> None:
        self._data = data

    def keys(self) -> Container:
        return self._data.columns

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Mapping[str, str]:
        return typing.cast(Mapping[str, str], self._data.iloc[idx])

    @classmethod
    def from_jsonl_file(cls, path: str | Path) -> DataFrameStorage:
        path = Path(path)

        assert path.is_file()

        with open(path) as f:
            lines = map(lambda s: s.strip("\n"), f.readlines())

        data = [json.loads(line) for line in lines]
        return cls.from_jsonl(data)

    @classmethod
    def from_jsonl(cls, data: Sequence[Mapping[str, str]]) -> DataFrameStorage:
        df = DataFrame.from_records(data)
        return cls(df)
