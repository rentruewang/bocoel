from typing import Protocol

from bocoel.corpora.interfaces import Embedder, Index, Storage


# TODO: Currently only supports 1 index for 1 single key.
# Maybe extend to support multiple indices?
class Corpus(Protocol):
    """
    Corpus is the entry point to handling the data in this library.

    A corpus has 3 main components:
    - Index: Indexes one particular column in the storage.Provides fast retrival.
    - Storage: Used to store the questions / answers / texts.
    - Embedder: Embeds the text into vectors for faster access.
    """

    index: Index
    """
    Index indexes one particular column in the storage into vectors.
    """

    storage: Storage
    """
    Storage is used to store the questions / answers / etc.
    Can be viewed as a dataframe of texts.
    """

    embedder: Embedder
    """
    Embedder is used to embed the texts into vectors.
    It should provide the ranges such that the index can be built.
    """
