# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from numpy.typing import NDArray

from bocoel.corpora.embedders import Embedder
from bocoel.corpora.indices import Index
from bocoel.corpora.storages import Storage


@dcls.dataclass(frozen=True)
class Corpus:
    """
    Corpus is the entry point to handling the data in this library.

    A corpus has 3 main components:
    - Index: Searches one particular column in the storage.Provides fast retrival.
    - Storage: Used to store the questions / answers / texts.
    - Embedder: Embeds the text into vectors for faster access.

    An index only corresponds to one key. If search over multiple keys is desired,
    a new column or a new corpus (with shared storage) should be created.
    """

    index: Index
    storage: Storage

    @classmethod
    def index_storage(
        cls,
        storage: Storage,
        embedder: Embedder,
        keys: Sequence[str],
        index_backend: type[Index],
        concat: Callable[[Iterable[Any]], str] = " [SEP] ".join,
        **index_kwargs: Any,
    ) -> "Corpus":
        """
        Creates a corpus from the given storage, embedder, key and index class,
        where storage entries would be mapped to strings,

        Parameters:
            storage: The storage to index.
            embedder: The embedder to use.
            keys: The keys to use for the index.
            index_backend: The index class to use.
            concat: The function to use to concatenate the keys.
            **index_kwargs: Additional arguments to pass to the index class.

        Returns:
            The created corpus.
        """

        def transform(mapping: Mapping[str, Sequence[Any]]) -> Sequence[str]:
            data = [mapping[k] for k in keys]
            return [concat(datum) for datum in zip(*data)]

        return cls.index_mapped(
            storage=storage,
            embedder=embedder,
            transform=transform,
            index_backend=index_backend,
            **index_kwargs,
        )

    @classmethod
    def index_mapped(
        cls,
        storage: Storage,
        embedder: Embedder,
        transform: Callable[[Mapping[str, Sequence[Any]]], Sequence[str]],
        index_backend: type[Index],
        **index_kwargs: Any,
    ) -> "Corpus":
        """
        Creates a corpus from the given storage, embedder, key and index class,
        where storage entries would be mapped to strings,
        using the specified batched transform function.

        Parameters:
            storage: The storage to index.
            embedder: The embedder to use.
            transform: The function to use to transform the storage entries.
            index_backend: The index class to use.
            **index_kwargs: Additional arguments to pass to the index class.

        Returns:
            The created corpus.
        """

        embeddings = embedder.encode_storage(storage, transform=transform)
        return cls.index_embeddings(
            embeddings=embeddings,
            storage=storage,
            index_backend=index_backend,
            **index_kwargs,
        )

    @classmethod
    def index_embeddings(
        cls,
        storage: Storage,
        embeddings: NDArray,
        index_backend: type[Index],
        **index_kwargs: Any,
    ) -> "Corpus":
        """
        Create the corpus with the given embeddings.
        This can be used to save time by encoding once and caching embeddings.

        Parameters:
            storage: The storage to use.
            embeddings: The embeddings to use.
            index_backend: The index class to use.
            **index_kwargs: Additional arguments to pass to the index class.

        Returns:
            The created corpus.
        """

        index = index_backend(embeddings, **index_kwargs)
        return cls(index=index, storage=storage)
