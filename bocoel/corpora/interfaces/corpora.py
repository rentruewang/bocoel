from typing import Protocol

from .searchers import Searcher
from .storages import Storage


class Corpus(Protocol):
    """
    Corpus is the entry point to handling the data in this library.

    A corpus has 3 main components:
    - Searcher: Searches one particular column in the storage.Provides fast retrival.
    - Storage: Used to store the questions / answers / texts.
    - Embedder: Embeds the text into vectors for faster access.

    A searcher only corresponds to one key. If search over multiple keys is desired,
    a new column or a new corpus (with shared storage) should be created.
    """

    storage: Storage
    """
    Storage is used to store the questions / answers / etc.
    Can be viewed as a dataframe of texts.
    """

    searcher: Searcher
    """
    Searcher searches one particular column in the storage into vectors.
    """
