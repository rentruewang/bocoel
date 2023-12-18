from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import ujson as json
from numpy.typing import NDArray

from bocoel.corpora.interfaces import Storage


# FIXME: Only supports file not directory for jsonl loading for now.
class JsonLStorage(Storage):
    def __init__(self, data: Sequence[Mapping[str, str]]) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(
        self, idx: int | slice | Sequence[int] | NDArray
    ) -> Mapping[str, str] | Sequence[Mapping[str, str]]:
        if isinstance(idx, (int, slice)):
            return self._data[idx]

        return [self._data[i] for i in idx]

    @classmethod
    def from_path(cls, path: str | Path) -> JsonLStorage:
        path = Path(path)

        assert path.is_file()

        with open(path) as f:
            lines = map(lambda s: s.strip("\n"), f.readlines())

        data = [json.loads(line) for line in lines]
        return cls(data)
