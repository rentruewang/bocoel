from bocoel import (
    ComposedCorpus,
    Corpus,
    DataFrameStorage,
    Distance,
    SBertEmbedder,
    WhiteningIndex,
)

from .indices import factories
from .storages import factories as storage_factories


def corpus(device: str) -> Corpus:
    storage = DataFrameStorage(storage_factories.df())
    embedder = SBertEmbedder(device=device)
    index_kwargs = {"distance": Distance.INNER_PRODUCT, **factories.whiten_kwargs()}
    return ComposedCorpus.index_storage(
        storage=storage,
        embedder=embedder,
        key="question",
        klass=WhiteningIndex,
        index_kwargs=index_kwargs,
    )
