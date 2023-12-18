from __future__ import annotations

import typing
from pathlib import Path
from typing import Mapping

import ujson as json
from pandas import DataFrame

from bocoel.corpora.interfaces import Storage


# FIXME: Only supports file not directory for jsonl loading now.
class JsonLStorage(Storage):
    def __init__(self, data: DataFrame) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Mapping[str, str]:
        return typing.cast(Mapping[str, str], self._data.iloc[idx])

    @classmethod
    def from_path(cls, path: str | Path) -> JsonLStorage:
        path = Path(path)

        assert path.is_file()

        with open(path) as f:
            lines = map(lambda s: s.strip("\n"), f.readlines())

        data = DataFrame.from_records([json.loads(line) for line in lines])
        return cls(data)
