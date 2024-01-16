from bocoel import (
    ComposedCorpus,
    Corpus,
    Distance,
    PandasStorage,
    SBertEmbedder,
    WhiteningIndex,
)
from tests import utils

from .indices import factories
from .storages import factories as storage_factories


@utils.cache
def corpus(device: str) -> Corpus:
    storage = PandasStorage(storage_factories.df())
    embedder = SBertEmbedder(device=device)
    return ComposedCorpus.index_storage(
        storage=storage,
        embedder=embedder,
        key="question",
        index_backend=WhiteningIndex,
        distance=Distance.INNER_PRODUCT,
        **factories.whiten_kwargs(),
    )
