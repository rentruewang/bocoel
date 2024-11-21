# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from pandas import DataFrame

from bocoel.corpora.storages.interfaces import Storage


class PandasStorage(Storage):
    """
    Storage for pandas DataFrame.
    Since pandas DataFrames are in-memory, this storage is fast,
    but might be memory inefficient and require a lot of RAM.
    """

    def __init__(self, df: DataFrame, /) -> None:
        self._df = df

    def keys(self) -> Collection[str]:
        return self._df.columns

    def __len__(self) -> int:
        return len(self._df)

    def _getitem(self, idx: int) -> Mapping[str, Any]:
        return self._df.iloc[idx].to_dict()

    @classmethod
    def from_jsonl_file(cls, path: str | Path, /) -> "PandasStorage":
        """
        Load data from a JSONL file.

        Parameters:
            path: The path to the file.

        Returns:
            A `PandasStorage` instance.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        if not path.is_file():
            raise ValueError(f"Cannot open file: {path}")

        with open(path) as f:
            lines = map(lambda s: s.strip("\n"), f.readlines())

        data = [json.loads(line) for line in lines]
        return cls.from_jsonl(data)

    @classmethod
    def from_jsonl(cls, data: Sequence[Mapping[str, str]], /) -> "PandasStorage":
        """
        Load data from a JSONL object or a list of JSON.

        Parameters:
            data: The JSONL object or list of JSON.

        Returns:
            A `PandasStorage` instance.
        """

        df = DataFrame.from_records(data)
        return cls(df)
