from bocoel import (
    ComposedCorpus,
    Corpus,
    Distance,
    Embedder,
    PandasStorage,
    WhiteningIndex,
)

from .indices import factories
from .storages import factories as storage_factories


def corpus(embedder: Embedder) -> Corpus:
    storage = PandasStorage(storage_factories.df())
    return ComposedCorpus.index_storage(
        storage=storage,
        embedder=embedder,
        keys=["question"],
        index_backend=WhiteningIndex,
        distance=Distance.INNER_PRODUCT,
        **factories.whiten_kwargs(),
    )
