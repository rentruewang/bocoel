# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
from collections import OrderedDict
from collections.abc import Mapping
from typing import Protocol

import numpy as np
import structlog
from numpy.typing import ArrayLike, NDArray

from bocoel.corpora import SearchResult

LOGGER = structlog.get_logger()


class SearchEvaluator(Protocol):
    """
    A protocol for evaluating the search results.
    """

    @abc.abstractmethod
    def __call__(self, results: Mapping[int, SearchResult], /) -> Mapping[int, float]:
        """
        Evaluates the given batched search result.
        The order of the results must be kept in the original order.

        Parameters:
            results: The results of the search. Mapping from index to search result.

        Returns:
            The results of the search. Must be in the same order as the query.
        """

        ...


class QueryEvaluator(Protocol):
    """
    A protocol for evaluating the query results.
    """

    @abc.abstractmethod
    def __call__(self, query: ArrayLike, /) -> OrderedDict[int, float]:
        """
        Evaluates the given batched query.
        The order of the results must be kept in the original order.

        Parameters:
            query: The query to evaluate.

        Returns:
            The results of the query. Must be in the same order as the query.
        """

        ...


class IndexEvaluator(Protocol):
    """
    A protocol for evaluating with the indices.
    """

    @abc.abstractmethod
    def __call__(self, idx: ArrayLike, /) -> NDArray:
        """
        Evaluates the given batched query.
        The order of the results must be kept in the original order.

        Parameters:
            idx: The indices to evaluate. Must be a 1D array.

        Returns:
            The indices of the results. Must be in the same order as the query.
        """

        ...


class CachedIndexEvaluator(IndexEvaluator):
    """
    Since there might be duplicate indices (and a lot of them during evaluation),
    this utility evaluator would cache the results and only compute the unseen indices.
    This would help evaluating the larger models of evaluation a lot faster.
    """

    def __init__(self, index_eval: IndexEvaluator, /) -> None:
        LOGGER.warning("Caching index evaluation. This may use a lot of memory.")

        self._index_eval = index_eval
        self._cache: dict[int, NDArray] = {}

    def __call__(self, idx: ArrayLike, /) -> NDArray:
        idx = np.array(idx)

        if idx.ndim != 1:
            raise ValueError(f"Expected 1D array, got {idx.ndim}D")

        # Only compute the previously unseen indices.
        unseen = [i for i in idx if i not in self._cache]

        if unseen:
            results = self._index_eval(unseen)
            mapped_results = dict(zip(unseen, results))
            self._cache |= mapped_results

        return np.array([self._cache[i] for i in idx])
