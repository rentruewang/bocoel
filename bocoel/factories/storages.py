from collections.abc import Mapping, Sequence
from typing import Any

from bocoel import ConcatStorage, DatasetsStorage, PandasStorage, Storage
from bocoel.common import StrEnum

from . import common


class StorageName(StrEnum):
    PANDAS = "PANDAS"
    DATASETS = "DATASETS"


@common.correct_kwargs
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
    storage = StorageName.lookup(storage)
    match storage:
        case StorageName.PANDAS:
            return common.correct_kwargs(PandasStorage.from_jsonl_file)(path)
        case StorageName.DATASETS:
            return common.correct_kwargs(DatasetsStorage.load)(
                path=path, name=name, split=split
            )
