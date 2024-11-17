# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from collections import OrderedDict
from collections.abc import Callable

import numpy as np
import structlog
from numpy.typing import NDArray

from bocoel.common import StrEnum
from bocoel.core.exams.interfaces import Exam
from bocoel.corpora import Index

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

    def _run(self, index: Index, results: OrderedDict[int, float]) -> NDArray:
        LOGGER.info("Running Accumulation exam", num_results=len(results))

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


def _check_dim(array: NDArray, /, ndim: int) -> None:
    if array.ndim != ndim:
        raise ValueError(f"Expected {ndim}D array, got {array.ndim}D")
