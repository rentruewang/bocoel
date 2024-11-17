# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from collections.abc import Collection, Mapping
from typing import Any

from bocoel.corpora.storages.interfaces import Storage


class DatasetsStorage(Storage):
    """
    Storage for datasets from HuggingFace Datasets library.
    Datasets are loaded on disk, so they might be slow(er) to load,
    but are more memory efficient.
    """

    def __init__(
        self, path: str, name: str | None = None, split: str | None = None
    ) -> None:
        # Optional dependency.
        import datasets
        from datasets import DatasetDict

        self._path = path
        self._name = name
        self._split = split

        ds = datasets.load_dataset(path=path, name=name, trust_remote_code=True)

        if split:
            if not isinstance(ds, DatasetDict):
                raise ValueError("Split is not supported for this dataset")

            ds = ds[split]

        self._dataset = ds

    def __repr__(self) -> str:
        args = [self._path, self._name, self._split, list(self.keys()), len(self)]

        args_str = ", ".join([str(arg) for arg in args if arg is not None])
        return f"Datasets({args_str})"

    def keys(self) -> Collection[str]:
        return self._dataset.column_names

    def __len__(self) -> int:
        return len(self._dataset)

    def _getitem(self, idx: int) -> Mapping[str, Any]:
        return self._dataset[idx]
