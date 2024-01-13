from enum import Enum

from bocoel import FaissIndex, HnswlibIndex, Index, PolarIndex, WhiteningIndex


class IndexName(str, Enum):
    FAISS = "faiss"
    HNSWLIB = "hnswlib"
    POLAR = "polar"
    WHITENING = "whitening"


def index_class_factory(name: str | IndexName, /) -> type[Index]:
    name = IndexName(name)

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
