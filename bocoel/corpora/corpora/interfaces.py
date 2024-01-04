from typing import Protocol

from bocoel.corpora.indices import Index
from bocoel.corpora.storages import Storage


class Corpus(Protocol):
    """
    Corpus is the entry point to handling the data in this library.

    A corpus has 3 main components:
    - Index: Searches one particular column in the storage.Provides fast retrival.
    - Storage: Used to store the questions / answers / texts.
    - Embedder: Embeds the text into vectors for faster access.

    An index only corresponds to one key. If search over multiple keys is desired,
    a new column or a new corpus (with shared storage) should be created.
    """

    storage: Storage
    """
    Storage is used to store the questions / answers / etc.
    Can be viewed as a dataframe of texts.
    """

    index: Index
    """
    Index searches one particular column in the storage into vectors.
    """
