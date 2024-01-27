from collections.abc import Collection, Mapping, Sequence
from typing import Any

import datasets
from datasets import Dataset, DatasetDict
from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.corpora.storages.interfaces import Storage


class DatasetsStorage(Storage):
    """
    Storage for datasets from HuggingFace Datasets library.
    Datasets are loaded on disk, so they might be slow(er) to load,
    but are more memory efficient.
    """

    def __init__(self, dataset: Dataset, /) -> None:
        self._dataset = dataset

    def keys(self) -> Collection[str]:
        return self._dataset.column_names

    def __len__(self) -> int:
        return len(self._dataset)

    def _getitem(self, idx: int) -> Mapping[str, Any]:
        return self._dataset[idx]

    @classmethod
    def load(cls, path: str, name: str, split: str = "") -> Self:
        ds = datasets.load_dataset(path=path, name=name)

        if split:
            if not isinstance(ds, DatasetDict):
                raise ValueError("Split is not supported for this dataset")

            ds = ds[split]

        return cls(ds)
