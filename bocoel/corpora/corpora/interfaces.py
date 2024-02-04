from typing import Protocol

from bocoel import common
from bocoel.corpora.indices import StatefulIndex
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

    index: StatefulIndex
    """
    Index searches one particular column in the storage into vectors.
    """

    def __repr__(self) -> str:
        name = common.remove_base_suffix(self, Corpus)
        return f"{name}({str(self.storage)}, {str(self.index)})"
