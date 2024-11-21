# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import NamedTuple

from numpy.typing import NDArray


@dcls.dataclass(frozen=True)
class _SearchResult:
    query: NDArray
    """
    Query vector.
    If batched, should have shape [batch, dims].
    Or else, should have shape [dims].
    """

    vectors: NDArray
    """
    Nearest neighbors.
    If batched, should have shape [batch, k, dims].
    Or else, should have shape [k, dims].
    """

    distances: NDArray
    """
    Calculated distance.
    If batched, should have shape [batch, k].
    Or else, should have shape [k].
    """

    indices: NDArray
    """
    Index in the original embeddings. Must be integers.
    If batched, should have shape [batch, k].
    Or else, should have shape [k].
    """


@dcls.dataclass(frozen=True)
class SearchResultBatch(_SearchResult):
    """
    A batched version of search result.
    """

    def __post_init__(self) -> None:
        if self.query.ndim != 2:
            raise ValueError(f"Query should be batched. Got shape {self.query.shape}")

        if self.vectors.ndim != 3:
            raise ValueError(
                f"Vectors should be batched. Got shape {self.vectors.shape}."
            )

        if self.distances.ndim != 2:
            raise ValueError(
                f"Distances should be batched. Got shape {self.distances.shape}."
            )

        if self.indices.ndim != 2:
            raise ValueError(
                f"Indices should be batched. Got shape {self.indices.shape}."
            )

        batches = {
            self.query.shape[0],
            self.vectors.shape[0],
            self.distances.shape[0],
            self.indices.shape[0],
        }

        if len(batches) != 1:
            raise ValueError(
                "Batched results should have the same batch size. "
                f"Got {len(self.query)}, {len(self.vectors)}, "
                f"{len(self.distances)}, {len(self.indices)}."
            )

        ks = {
            self.vectors.shape[1],
            self.distances.shape[1],
            self.indices.shape[1],
        }

        if len(ks) != 1:
            raise ValueError(
                "Batched results should have the same number of neighbors. "
                f"Got {self.vectors.shape[1]}, {self.distances.shape[1]}, "
                f"{self.indices.shape[1]}."
            )


@dcls.dataclass(frozen=True)
class SearchResult(_SearchResult):
    """
    A non-batched version of search result.
    """

    def __post_init__(self) -> None:
        if self.query.ndim != 1:
            raise ValueError(
                f"Query should not be batched. Got shape {self.query.shape}."
            )

        if self.vectors.ndim != 2:
            raise ValueError(
                f"Vectors should not be batched. Got shape {self.vectors.shape}."
            )

        if self.distances.ndim != 1:
            raise ValueError(
                f"Distances should not be batched. Got shape {self.distances.shape}."
            )

        if self.indices.ndim != 1:
            raise ValueError(
                f"Indices should not be batched. Got shape {self.indices.shape}."
            )

        ks = {
            self.vectors.shape[0],
            self.distances.shape[0],
            self.indices.shape[0],
        }

        if len(ks) != 1:
            raise ValueError(
                "Non-batched results should have the same number of neighbors. "
                f"Got {self.vectors.shape[0]}, {self.distances.shape[0]}, "
                f"{self.indices.shape[0]}."
            )


class InternalResult(NamedTuple):
    distances: NDArray
    """
    Calculated distance.
    """

    indices: NDArray
    """
    Index in the original embeddings. Must be integers.
    """
