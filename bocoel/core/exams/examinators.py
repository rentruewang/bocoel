from collections import OrderedDict
from collections.abc import Mapping

from pandas import DataFrame
from typing_extensions import Self

from bocoel.core.exams.interfaces import Exam
from bocoel.corpora import Index

from .columns import exams
from .stats import AccType, Accumulation


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

        TODO:
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
    def presets(cls) -> Self:
        """
        Returns:
            The default examinator.
        """

        return cls(
            {
                exams.ACC_MIN: Accumulation(AccType.MIN),
                exams.ACC_MAX: Accumulation(AccType.MAX),
                exams.ACC_AVG: Accumulation(AccType.AVG),
            }
        )
