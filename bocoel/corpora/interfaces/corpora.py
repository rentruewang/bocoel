from __future__ import annotations

import abc
from typing import Container, Mapping, Protocol, Sequence

from numpy.typing import NDArray

from bocoel.corpora.interfaces import Embedder, Index, Storage


class Corpus(Protocol):
    index: Index
    storage: Storage
    embedder: Embedder
