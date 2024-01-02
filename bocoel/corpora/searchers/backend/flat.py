import heapq
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, device
from typing_extensions import Self

from bocoel.corpora.interfaces import Distance, Searcher, SearchResult
from bocoel.corpora.searchers import utils

Device = str | device


# TODO: Add tests.
class FlatSearcher(Searcher):
    def __init__(
        self,
        embeddings: NDArray,
        distance: str | Distance,
        device: Device,
        batch_size: int = -1,
    ) -> None:
        utils.validate_embeddings(embeddings)
        embeddings = utils.normalize(embeddings)
        self._emb = utils.normalize(embeddings)
        self._bounds = utils.boundaries(embeddings)
        self._dims = embeddings.shape[1]
        self._distance = Distance(distance)

        # Public attributes because they can be changed at anytime.
        self.device = device
        self.batch_size = batch_size

    def _get_distance(self) -> Distance:
        return self._distance

    def _set_distance(self, dist: Distance) -> None:
        self._distance = dist

    distance = property(_get_distance, _set_distance)

    @property
    def embeddings(self) -> NDArray:
        return self._emb

    @property
    def bounds(self) -> NDArray:
        return self._bounds

    @property
    def dims(self) -> int:
        return self._emb.shape[1]

    def _search(self, query: NDArray, k: int = 1) -> SearchResult:
        # Copying to prevent batch_size changed mid computation.
        batch_size = self.batch_size if self.batch_size > 0 else len(self._emb)
        query_torch = torch.tensor(query, device=self.device)

        match self._distance:
            case Distance.L2:
                dist_fn = self._l2
                top_k_fn = heapq.nsmallest
            case Distance.INNER_PRODUCT:
                dist_fn = self._inner_product
                top_k_fn = heapq.nlargest

        dists = []
        for idx in range(0, len(self._emb), batch_size):
            batch = self._emb[idx : idx + batch_size]
            batch_torch = torch.tensor(batch, device=self.device)
            dists.extend(dist_fn(batch=batch_torch, query=query_torch))

        # Adapted from here.
        # https://stackoverflow.com/a/18691983
        top_k_idx = np.array(top_k_fn(k, range(len(dists)), key=dists.__getitem__))
        top_k_scores = np.array([dists[i] for i in top_k_idx])

        return SearchResult(
            vectors=self._emb[top_k_idx], scores=top_k_scores, indices=top_k_idx
        )

    @staticmethod
    def _inner_product(batch: Tensor, query: Tensor) -> Tensor:
        return batch @ query

    @staticmethod
    def _l2(batch: Tensor, query: Tensor) -> Tensor:
        diff = batch - query[None, :]
        return (diff**2).sum(dim=1)

    @classmethod
    def from_embeddings(
        cls, embeddings: NDArray, distance: str | Distance, **kwargs: Any
    ) -> Self:
        return cls(embeddings=embeddings, distance=distance, **kwargs)
