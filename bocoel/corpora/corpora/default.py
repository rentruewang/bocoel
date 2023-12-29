import dataclasses as dcls

from bocoel.corpora.interfaces import Corpus, Embedder, Searcher, Storage


@dcls.dataclass(frozen=True)
class ComposedCorpus(Corpus):
    """
    Simply a collection of components.
    """

    searcher: Searcher
    storage: Storage
    embedder: Embedder
