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
