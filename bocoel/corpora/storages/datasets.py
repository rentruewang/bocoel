from collections.abc import Collection, Mapping, Sequence
from typing import Any

from datasets import Dataset

from bocoel.corpora.storages.interfaces import Storage


class DatasetsStorage(Storage):
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    def keys(self) -> Collection[str]:
        return self._dataset.column_names

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        return self._dataset[idx]

    def get(self, key: str) -> Sequence[Any]:
        if not isinstance(key, str):
            raise ValueError("Key should be a string")

        return self._dataset[key]
