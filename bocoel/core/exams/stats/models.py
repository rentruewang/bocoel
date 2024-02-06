from collections import OrderedDict
from collections.abc import Callable
from typing import Literal

import networkx as nx
import numpy as np
import structlog
from numpy.typing import NDArray
from scipy.spatial import distance

from bocoel.common import StrEnum
from bocoel.core.exams.interfaces import Exam
from bocoel.corpora import StatefulIndex

LOGGER = structlog.get_logger()


class AccType(StrEnum):
    """
    Accumulation type.
    """

    MIN = "MINIMUM"
    "Minimum value accumulation."

    MAX = "MAXIMUM"
    "Maximum value accumulation."

    AVG = "AVERAGE"
    "Average value accumulation."


class Accumulation(Exam):
    """
    Accumulation is an exam designed to evaluate the min / max / avg of the history.
    """

    def __init__(self, typ: AccType) -> None:
        self._acc_func: Callable[[NDArray], NDArray]
        match typ:
            case AccType.MIN:
                self._acc_func = np.minimum.accumulate
            case AccType.MAX:
                self._acc_func = np.maximum.accumulate
            case AccType.AVG:
                self._acc_func = lambda arr: np.cumsum(arr) / np.arange(1, arr.size + 1)
            case _:
                raise ValueError(f"Unknown accumulation type {typ}")

    def _run(self, index: StatefulIndex, results: OrderedDict[int, float]) -> NDArray:
        _ = index

        values = np.array(list(results.values()))
        return self._acc(values, self._acc_func)

    @staticmethod
    def _acc(array: NDArray, accumulate: Callable[[NDArray], NDArray]) -> NDArray:
        """
        Accumulate the array using the given function.

        Parameters:
            array: The array to accumulate.
            accumulate: The accumulation function to use.

        Returns:
            The accumulated array.

        Raises:
            ValueError: If the array is not 1D.
        """

        _check_dim(array, 1)
        result = accumulate(array)
        _check_dim(result, 1)
        return result


class MstMaxEdgeType(StrEnum):
    QUERY = "QUERY"
    DATA = "DATA"


class MstMaxEdge(Exam):
    """
    MstMaxEdge is an exam designed to evaluate the maximum edge of the minimum spanning tree.
    This can be thought of as the smallest density of a density based method.
    """

    def __init__(self, typ: MstMaxEdgeType) -> None:
        self._agg_type = typ

    def _run(self, index: StatefulIndex, results: OrderedDict[int, float]) -> NDArray:
        # TODO: Only supports L2 for now. Would like more options.
        points = self._points(index=index, results=results)
        return self._max_mst_edge_acc(points, metric="euclidean")

    def _points(
        self, index: StatefulIndex, results: OrderedDict[int, float]
    ) -> NDArray:

        match self._agg_type:
            case MstMaxEdgeType.QUERY:
                return np.array([index[idx].query for idx in results.keys()])
            case MstMaxEdgeType.DATA:
                return np.concatenate([index[idx].vectors for idx in results.keys()])
            case _:
                raise ValueError(f"Unknown aggregation type {self._agg_type}")

    @staticmethod
    def _max_mst_edge_acc(
        array: NDArray, metric: Literal["euclidean"] = "euclidean"
    ) -> NDArray:
        WEIGHT = "weight"
        DUMMY = 1

        _check_dim(array, 2)

        results = [float("inf")]
        for nodes in range(2, len(array) + 1):
            # NOTE:
            # Add some dummy value in case there are samples in the corner region
            # which would result in truncating to the same points.
            # Since networkx uses sparse matrix, 0 cost would create a disjoint graph.
            dists = distance.pdist(array[:nodes], metric=metric) + DUMMY
            graph = nx.from_numpy_array(distance.squareform(dists))
            edges = list(nx.minimum_spanning_edges(graph))

            assert (
                len(edges) == nodes - 1
            ), f"MST should have n-1 edges. Got {len(edges)} instead of {nodes - 1}"

            source, target, weight = max(edges, key=lambda x: x[2][WEIGHT])
            LOGGER.debug(
                "max_mst_edge_acc", source=source, target=target, weight=weight[WEIGHT]
            )

            results.append(weight[WEIGHT] - DUMMY)
        return np.array(results)


def _check_dim(array: NDArray, /, ndim: int) -> None:
    if array.ndim != ndim:
        raise ValueError(f"Expected {ndim}D array, got {array.ndim}D")
