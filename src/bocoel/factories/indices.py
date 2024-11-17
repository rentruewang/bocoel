# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from typing import Any

from bocoel import FaissIndex, HnswlibIndex, Index, PolarIndex, WhiteningIndex
from bocoel.common import ItemNotFound, StrEnum


class IndexName(StrEnum):
    """
    The names of the indices.
    """

    FAISS = "FAISS"
    "Corresponds to `FaissIndex`."

    HNSWLIB = "HNSWLIB"
    "Corresponds to `HnswlibIndex`."

    POLAR = "POLAR"
    "Corresponds to `PolarIndex`."

    WHITENING = "WHITENING"
    "Corresponds to `WhiteningIndex`."


def index_class(name: str | IndexName, /) -> type[Index]:
    """
    Get the index class for the given name.

    Parameters:
        name: The name of the index.
    """

    name = IndexName.lookup(name)

    match name:
        case IndexName.FAISS:
            return FaissIndex
        case IndexName.HNSWLIB:
            return HnswlibIndex
        case IndexName.POLAR:
            return PolarIndex
        case IndexName.WHITENING:
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
                idx = IndexName.lookup(value)
                mapped[key] = index_class(idx)
        except ItemNotFound:
            pass

    return mapped
