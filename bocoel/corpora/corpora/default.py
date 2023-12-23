from __future__ import annotations

import dataclasses as dcls
from typing import Container, Mapping, Sequence

from numpy.typing import NDArray

from bocoel.corpora.interfaces import Corpus, Embedder, Index, Storage


@dcls.dataclass(frozen=True)
class ComposedCorpus(Corpus):
    index: Index
    storage: Storage
    embedder: Embedder
