from collections import OrderedDict
from typing import Literal

import networkx as nx
import numpy as np
import structlog
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.spatial import distance

from bocoel.common import StrEnum
from bocoel.corpora import StatefulIndex

WEIGHT = "weight"


class MetricType(StrEnum):
    VALUE = "VALUE"
    "VALUE is for retaining the original value. Does not calculate anything."

    ACC_MIN = "ACC_MIN"
    "ACC_MIN is for calculating the minimum so far at each step."

    ACC_MAX = "ACC_MAX"
    "ACC_MAX is for calculating the maximum so far at each step."

    ACC_AVG = "ACC_AVG"
    "ACC_AVG is for calculating the average so far at each step."

    MAX_MST_QUERY = "MAX_MST_EDGE_QUERY"
    "MAX_MST_QUERY is for calculating the how disjointed for the mst formed by given query."

    MAX_MST_DATA = "MAX_MST_EDGE_DATA"
    "MAX_MST_DATA is for calculating the how disjointed for the mst formed by retrieved data."


LOGGER = structlog.get_logger()


def stats(index: StatefulIndex, states: OrderedDict[int, float]) -> DataFrame:
    return DataFrame.from_dict(
        {
            metric: _stats_per_metric(name=metric, index=index, states=states)
            for metric in MetricType
        }
    )


def _stats_per_metric(
    name: MetricType, index: StatefulIndex, states: OrderedDict[int, float]
) -> NDArray:
    if name is MetricType.VALUE:
        return np.array(list(states.values()))

    match name:
        case MetricType.ACC_MIN | MetricType.ACC_MAX | MetricType.ACC_AVG:
            array = np.array(list(states.values()))
        case MetricType.MAX_MST_QUERY:
            history = index.history
            array = np.array([history[idx].query for idx in states.keys()])
        case MetricType.MAX_MST_DATA:
            history = index.history
            array = np.concatenate([history[idx].vectors for idx in states.keys()])
        case _:
            raise ValueError(f"Unknown metric {name}")

    match name:
        case MetricType.ACC_MIN:
            return acc_min(array)
        case MetricType.ACC_MAX:
            return acc_max(array)
        case MetricType.ACC_AVG:
            return acc_avg(array)
        case MetricType.MAX_MST_QUERY | MetricType.MAX_MST_DATA:
            return max_mst_edge_acc(array)
        case _:
            raise ValueError(f"Unknown metric {name}")


def _check_dim(array: NDArray, /, ndim: int) -> None:
    if array.ndim != ndim:
        raise ValueError(f"Expected {ndim}D array, got {array.ndim}D")


def acc_min(array: NDArray) -> NDArray:
    _check_dim(array, 1)
    result = np.minimum.accumulate(array)
    _check_dim(result, 1)
    return result


def acc_max(array: NDArray) -> NDArray:
    _check_dim(array, 1)
    result = np.maximum.accumulate(array)
    _check_dim(result, 1)
    return result


def acc_avg(array: NDArray) -> NDArray:
    _check_dim(array, 1)
    result = np.cumsum(array) / np.arange(1, array.size + 1)
    _check_dim(result, 1)
    return result


def max_mst_edge_acc(
    array: NDArray, metric: Literal["euclidean"] = "euclidean"
) -> NDArray:
    _check_dim(array, 2)

    results = [float("inf")]
    for nodes in range(2, len(array) + 1):
        dists = distance.pdist(array[:nodes], metric=metric)
        graph = nx.from_numpy_array(distance.squareform(dists))
        source, target, weight = max(
            nx.minimum_spanning_edges(graph), key=lambda x: x[2][WEIGHT]
        )
        LOGGER.debug(
            "max_mst_edge_acc", source=source, target=target, weight=weight[WEIGHT]
        )
        results.append(weight[WEIGHT])
    return np.array(results)
