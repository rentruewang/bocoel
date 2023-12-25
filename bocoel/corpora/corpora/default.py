import dataclasses as dcls

from bocoel.corpora.interfaces import Corpus, Embedder, Index, Storage


@dcls.dataclass(frozen=True)
class ComposedCorpus(Corpus):
    index: Index
    storage: Storage
    embedder: Embedder
