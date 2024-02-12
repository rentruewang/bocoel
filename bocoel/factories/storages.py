from collections.abc import Mapping, Sequence
from typing import Any

from bocoel import ConcatStorage, DatasetsStorage, PandasStorage, Storage
from bocoel.common import StrEnum

from . import common


class StorageName(StrEnum):
    """
    The storage names.
    """

    PANDAS = "PANDAS"
    "Corresponds to `PandasStorage`."

    DATASETS = "DATASETS"
    "Corresponds to `DatasetsStorage`."


@common.correct_kwargs
def storage(names: Sequence[str], /, configs: Sequence[Mapping[str, Any]]) -> Storage:
    """
    Create a storage.

    Parameters:
        names: The names of the storages.
        configs: The configurations for the storages.

    Returns:
        The storage instance.

    Raises:
        ValueError: If the names and configs do not have the same length.
        ValueError: If the storage is unknown.
    """

    if len(names) != len(configs):
        raise ValueError("Names and configs must have the same length")

    storages = [
        _storage_factory_single(name, **config) for name, config in zip(names, configs)
    ]

    return ConcatStorage.join(storages)


def _storage_factory_single(
    storage: str | StorageName, /, *, path: str, name: str, split: str
) -> Storage:
    """
    Create a single storage.

    Parameters:
        storage: The name of the storage.
        path: The path to the storage.
        name: The name of the storage.
        split: The split to use.

    Returns:
        The storage instance.

    Raises:
        ValueError: If the storage is unknown.
    """

    storage = StorageName.lookup(storage)
    match storage:
        case StorageName.PANDAS:
            return common.correct_kwargs(PandasStorage.from_jsonl_file)(path)
        case StorageName.DATASETS:
            return common.correct_kwargs(DatasetsStorage.load)(
                path=path, name=name, split=split
            )
        case _:
            raise ValueError(f"Unknown storage name {storage}")
