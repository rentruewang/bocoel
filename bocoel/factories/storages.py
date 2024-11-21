# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from bocoel import DatasetsStorage, PandasStorage, Storage
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
def storage(
    storage: str | StorageName, /, *, path: str = "", name: str = "", split: str = ""
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
            return common.correct_kwargs(DatasetsStorage)(
                path=path, name=name, split=split
            )
        case _:
            raise ValueError(f"Unknown storage name {storage}")
