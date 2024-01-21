from collections.abc import Callable, Iterable

import numpy as np
from numpy import linalg
from numpy.typing import ArrayLike, NDArray

from .interfaces import Boundary, IndexedArray, SearchResult, SearchResultBatch


def validate_embeddings(
    embeddings: NDArray, /, allowed_ndims: int | list[int] = 2
) -> None:
    if isinstance(allowed_ndims, int):
        allowed_ndims = [allowed_ndims]

    if embeddings.ndim not in allowed_ndims:
        d_str = " or ".join(f"{i}D" for i in allowed_ndims)
        raise ValueError(f"Expected embeddings to be {d_str}, got {embeddings.ndim}D")


def normalize(embeddings: ArrayLike, /, p: int = 2) -> NDArray:
    embeddings = np.array(embeddings)
    validate_embeddings(embeddings, allowed_ndims=[1, 2])

    # Axis = -1 works for both 1D and 2D.
    norm = linalg.norm(embeddings, axis=-1, ord=p, keepdims=True)
    return embeddings / norm


def boundaries(embeddings: NDArray, /) -> Boundary:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings to be 2D, got {embeddings.ndim}D")

    return Boundary(np.stack([embeddings.min(axis=0), embeddings.max(axis=0)]).T)


class Indexer(IndexedArray):
    def __init__(
        self,
        embeddings: NDArray | IndexedArray,
        mapping: Callable[[NDArray], NDArray] = lambda x: x,
    ) -> None:
        self._emb = embeddings
        self._mapping = mapping

    def __len__(self) -> int:
        return len(self._emb)

    def __getitem__(self, index: int | NDArray, /) -> NDArray:
        if isinstance(index, int):
            item = self._emb[index]
            return self._mapping(item)

        items = [self._emb[k] for k in index]
        return np.array([self._mapping(item) for item in items])


def split_search_result_batch(srb: SearchResultBatch, /) -> list[SearchResult]:
    return [
        SearchResult(query=q, vectors=v, distances=d, indices=i)
        for q, v, d, i in zip(srb.query, srb.vectors, srb.distances, srb.indices)
    ]


def join_search_results(srs: Iterable[SearchResult], /) -> SearchResultBatch:
    return SearchResultBatch(
        query=np.stack([sr.query for sr in srs]),
        vectors=np.stack([sr.vectors for sr in srs]),
        distances=np.stack([sr.distances for sr in srs]),
        indices=np.stack([sr.indices for sr in srs]),
    )
