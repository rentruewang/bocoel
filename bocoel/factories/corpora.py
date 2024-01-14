from typing import Any

from bocoel import ComposedCorpus, Corpus, Embedder, Storage
from bocoel.common import StrEnum

from . import indices
from .indices import IndexName


class CorpusName(StrEnum):
    COMPOSED = "COMPOSED"


def corpus_factory(
    name: str | CorpusName = CorpusName.COMPOSED,
    /,
    *,
    storage: Storage,
    embedder: Embedder,
    index_name: str | IndexName,
    **index_kwargs: Any,
) -> Corpus:
    if CorpusName.lookup(name) is not CorpusName.COMPOSED:
        raise ValueError(f"Unknown corpus name: {name}")

    return ComposedCorpus.index_storage(
        storage=storage,
        embedder=embedder,
        index_backend=indices.index_class_factory(index_name),
        **index_kwargs,
    )
