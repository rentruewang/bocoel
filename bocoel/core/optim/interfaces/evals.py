import abc
from collections import OrderedDict
from collections.abc import Mapping
from typing import Protocol

from numpy.typing import ArrayLike

from bocoel.corpora import SearchResult


class SearchEvaluator(Protocol):
    @abc.abstractmethod
    def __call__(self, results: Mapping[int, SearchResult], /) -> Mapping[int, float]:
        """
        Evaluates the given batched search result.
        The order of the results must be kept in the original order.
        """

        ...


class QueryEvaluator(Protocol):
    @abc.abstractmethod
    def __call__(self, query: ArrayLike, /) -> OrderedDict[int, float]:
        """
        Evaluates the given batched query.
        The order of the results must be kept in the original order.
        """

        ...
