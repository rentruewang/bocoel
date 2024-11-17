# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
from pandas import DataFrame

from bocoel.core.exams.interfaces import Exam
from bocoel.corpora import Index

from .columns import exams
from .stats import Accumulation


class Examinator:
    """
    The examinator is responsible for launching exams.
    Examinators take in an index and results of an optimizer run,
    and return a DataFrame of scores for
    the accumulated history performance of the optimizer.
    """

    def __init__(self, exams: Mapping[str, Exam]) -> None:
        self.exams = exams

    def examine(self, index: Index, results: OrderedDict[int, float]) -> DataFrame:
        """
        Perform the exams on the results.
        This method looks up results in the index and runs the exams on the results.

        Parameters:
            index: The index of the results.
            results: The results.

        Returns:
            The scores of the exams.

        Todo:
            Run the different exams in parallel.
            Currently the exams are run sequentially and can be slow.
        """

        scores = {k: v.run(index, results) for k, v in self.exams.items()}
        original = {
            exams.STEP_IDX: list(range(len(results))),
            exams.ORIGINAL: list(results.values()),
        }
        return DataFrame.from_dict({**original, **scores})

    @classmethod
    def presets(cls) -> "Examinator":
        """
        Returns:
            The default examinator.
        """

        avg_acc = lambda arr: np.cumsum(arr) / np.arange(1, arr.size + 1)

        return cls(
            {
                exams.ACC_MIN: Accumulation(np.minimum.accumulate),
                exams.ACC_MAX: Accumulation(np.maximum.accumulate),
                exams.ACC_AVG: Accumulation(avg_acc),
            }
        )
