# Copyright (c) BoCoEL Authors - All Rights Reserved

from collections.abc import Sequence
from typing import Any

from bocoel import Corpus, Embedder, Storage

from . import indices

__all__ = ["corpus"]


def corpus(
    *,
    storage: Storage,
    embedder: Embedder,
    keys: Sequence[str],
    index_name: str,
    **index_kwargs: Any,
) -> Corpus:
    """
    Create a corpus.

    Parameters:
        name: The name of the corpus.
        storage: The storage to use.
        embedder: The embedder to use.
        keys: The key to use for the index.
        index_name: The name of the index backend to use.
        **index_kwargs: The keyword arguments to pass to the index backend.

    Returns:
        The corpus instance.

    Raises:
        ValueError: If the name is unknown.
    """

    return Corpus.index_storage(
        storage=storage,
        embedder=embedder,
        keys=keys,
        index_backend=indices.index_class(index_name),
        **indices.index_set_backends(index_kwargs),
    )
