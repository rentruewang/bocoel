from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from bocoel import ConcatStorage, DatasetsStorage, PandasStorage, Storage


class StorageName(str, Enum):
    PANDAS = "pandas"
    DATASETS = "datasets"


def storage_factory(
    names: Sequence[str], /, configs: Sequence[Mapping[str, Any]]
) -> Storage:
    if len(names) != len(configs):
        raise ValueError("Names and configs must have the same length")

    storages = [
        _storage_factory_single(name, **config) for name, config in zip(names, configs)
    ]

    return ConcatStorage.join(storages)


def _storage_factory_single(
    storage: str | StorageName, /, *, path: str, name: str, split: str
) -> Storage:
    storage = StorageName(storage)
    match storage:
        case StorageName.PANDAS:
            return PandasStorage.from_jsonl_file(path)
        case StorageName.DATASETS:
            return DatasetsStorage.load(path=path, name=name, split=split)
