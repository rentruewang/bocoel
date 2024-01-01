from collections.abc import Collection, Mapping, Sequence

from datasets import Dataset

from bocoel.corpora.interfaces import Storage


# TODO: Write tests for this class.
class HuggingfaceDatasetsStorage(Storage):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__()

        self._dataset = dataset

    def keys(self) -> Collection[str]:
        return self._dataset.column_names

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Mapping[str, str]:
        return self._dataset[idx]

    def get(self, key: str) -> Sequence[str]:
        if not isinstance(key, str):
            raise ValueError("Key should be a string")

        return self._dataset[key]
