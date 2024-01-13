from collections.abc import Collection, Mapping, Sequence
from typing import Any

import datasets
from datasets import Dataset, DatasetDict
from typing_extensions import Self

from bocoel.corpora.storages.interfaces import Storage


class DatasetsStorage(Storage):
    def __init__(self, dataset: Dataset, /) -> None:
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

    @classmethod
    def load(cls, path: str, name: str, split: str = "") -> Self:
        ds = datasets.load_dataset(path=path, name=name)

        if split:
            if not isinstance(ds, DatasetDict):
                raise ValueError("Split is not supported for this dataset")

            ds = ds[split]

        return cls(ds)
