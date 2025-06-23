# Copyright (c) BoCoEL Authors - All Rights Reserved

from bocoel import DatasetsStorage, PandasStorage, Storage

from . import common

__all__ = ["storage"]


@common.correct_kwargs
def storage(
    storage: str, /, *, path: str = "", name: str = "", split: str = ""
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

    match storage:
        case "PANDAS":
            return PandasStorage.from_jsonl_file(path)
        case "DATASETS":
            return DatasetsStorage(path=path, name=name, split=split)
        case _:
            raise ValueError(f"Unknown storage name {storage}")
