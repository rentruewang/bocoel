from typing import Protocol

from bocoel.corpora.interfaces import Embedder, Index, Storage


# TODO: Currently only supports 1 index for 1 single key.
# Maybe extend to support multiple indices?
class Corpus(Protocol):
    index: Index
    storage: Storage
    embedder: Embedder
