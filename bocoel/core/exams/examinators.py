from collections import OrderedDict
from collections.abc import Mapping

from pandas import DataFrame
from typing_extensions import Self

from bocoel.core.exams.interfaces import Exam
from bocoel.corpora import StatefulIndex

from .columns import exams
from .stats import AccType, Accumulation, MstMaxEdge, MstMaxEdgeType, Segregation


class Examinator:
    """
    The examinator is responsible for launching exams.
    Examinators take in an index and results of an optimizer run,
    and return a DataFrame of scores for
    the accumulated history performance of the optimizer.
    """

    def __init__(self, exams: Mapping[str, Exam]) -> None:
        self.exams = exams

    def examine(
        self, index: StatefulIndex, results: OrderedDict[int, float]
    ) -> DataFrame:
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
                exams.MST_MAX_EDGE_QUERY: MstMaxEdge(MstMaxEdgeType.QUERY),
                exams.MST_MAX_EDGE_DATA: MstMaxEdge(MstMaxEdgeType.DATA),
                exams.SEGREGATION: Segregation(),
            }
        )
