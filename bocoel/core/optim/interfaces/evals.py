import abc
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Protocol

from numpy.typing import ArrayLike

from bocoel.corpora import SearchResult


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


class IdEvaluator(Protocol):
    """
    A protocol for evaluating with the indices.
    """

    @abc.abstractmethod
    def __call__(self, idx: ArrayLike, /) -> Sequence[float]:
        """
        Evaluates the given batched query.
        The order of the results must be kept in the original order.

        Parameters:
            idx: The indices to evaluate.

        Returns:
            The indices of the results. Must be in the same order as the query.
        """

        ...
