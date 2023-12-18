from __future__ import annotations

import dataclasses as dcls

from corpora.interfaces import Corpus, Embedder, Index, Storage


@dcls.dataclass(frozen=True)
class SimpleCorpus(Corpus):
    index: Index
    storage: Storage
    embedder: Embedder
