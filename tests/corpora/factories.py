from bocoel import (
    ComposedCorpus,
    Corpus,
    DataFrameStorage,
    Distance,
    SBertEmbedder,
    WhiteningIndex,
)

from .indices import factories
from .storages import test_df_storage


def corpus(device: str) -> Corpus:
    storage = DataFrameStorage(test_df_storage.df())
    embedder = SBertEmbedder(device=device)
    index_kwargs = {"distance": Distance.INNER_PRODUCT, **factories.whiten_kwargs()}
    return ComposedCorpus.index_storage(
        storage=storage,
        embedder=embedder,
        key="question",
        klass=WhiteningIndex,
        index_kwargs=index_kwargs,
    )
