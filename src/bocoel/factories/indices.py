# Copyright (c) BoCoEL Authors - All Rights Reserved

from typing import Any

from bocoel import FaissIndex, HnswlibIndex, Index, PolarIndex, WhiteningIndex
from bocoel.common import ItemNotFound

__all__ = ["index_class", "index_set_backends"]


def index_class(name: str, /) -> type[Index]:
    """
    Get the index class for the given name.

    Parameters:
        name: The name of the index.
    """
    match name:
        case "FAISS":
            return FaissIndex
        case "HNSWLIB":
            return HnswlibIndex
        case "POLAR":
            return PolarIndex
        case "WHITENING":
            return WhiteningIndex
        case _:
            raise ValueError(f"Unknown index name: {name}")


def index_set_backends(kwargs: dict[str, Any], /) -> dict[str, Any]:
    """
    Sets the backend variable to the desired class in `kwargs`.

    Parameters:
        kwargs: The keyword arguments to map.

    Returns:
        The mapped keyword arguments.
    """

    mapped = {**kwargs}

    for key, value in kwargs.items():
        try:
            if isinstance(value, str):
                mapped[key] = index_class(value)
        except ItemNotFound:
            pass

    return mapped
