from typing import Any

from bocoel import FaissIndex, HnswlibIndex, Index, PolarIndex, WhiteningIndex
from bocoel.common import StrEnum


class IndexName(StrEnum):
    FAISS = "FAISS"
    HNSWLIB = "HNSWLIB"
    POLAR = "POLAR"
    WHITENING = "WHITENING"


def index_class_factory(name: str | IndexName, /) -> type[Index]:
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


def index_set_backend(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Sets the backend variable to the desired class in `kwargs`.
    """

    mapped = {**kwargs}
    BACKEND = "backend"

    if BACKEND in mapped:
        mapped[BACKEND] = index_class_factory(IndexName.lookup(mapped[BACKEND]))

    return mapped
