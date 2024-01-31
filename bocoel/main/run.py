from collections.abc import Mapping
from typing import Any

import bocoel
from bocoel import factories
from bocoel.factories import (
    AdaptorName,
    CorpusName,
    EmbedderName,
    IndexName,
    LMName,
    OptimizerName,
    StorageName,
)


def with_kwargs(
    embedder_name: str | EmbedderName,
    embedder_kwargs: Mapping[str, Any],
    index_name: str | IndexName,
    index_kwargs: Mapping[str, Any],
    storage_name: str | StorageName,
    storage_kwargs: Mapping[str, Any],
    corpus_name: str | CorpusName,
    adaptor_name: str | AdaptorName,
    adaptor_kwargs: Mapping[str, Any],
    lm_name: str | LMName,
    lm_kwargs: Mapping[str, Any],
    optimizer_name: str | OptimizerName,
    optimizer_kwargs: Mapping[str, Any],
    iterations: int,
):
    embedder = factories.embedder_factory(embedder_name, **embedder_kwargs)
    storage = factories.storage_factory(storage_name, **storage_kwargs)
    corpus = factories.corpus_factory(
        corpus_name,
        storage=storage,
        embedder=embedder,
        index_name=index_name,
        **index_kwargs,
    )
    lm = factories.lm_factory(lm_name, **lm_kwargs)
    adaptor = factories.adaptor_factory(adaptor_name, **adaptor_kwargs)
    optim = factories.optimizer_factory(
        optimizer_name, corpus=corpus, adaptor=adaptor, **optimizer_kwargs
    )
    bocoel.bocoel(optimizer=optim, iterations=iterations)
