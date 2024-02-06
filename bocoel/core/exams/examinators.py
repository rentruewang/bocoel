from collections import OrderedDict
from collections.abc import Mapping

from pandas import DataFrame
from typing_extensions import Self

from bocoel.common import StrEnum
from bocoel.core.exams.interfaces import Exam
from bocoel.corpora import StatefulIndex

from .stats import AccType, Accumulation, MstMaxEdge, MstMaxEdgeType, Segregation


class ExamName(StrEnum):
    ORIGINAL = "ORIGINAL"
    STEP_IDX = "STEP_IDX"
    ACC_MIN = "ACC_MIN"
    ACC_MAX = "ACC_MAX"
    ACC_AVG = "ACC_AVG"
    MST_MAX_EDGE_QUERY = "MST_MAX_EDGE_QUERY"
    MST_MAX_EDGE_DATA = "MST_MAX_EDGE_DATA"
    SEGREGATION = "SEGREGATION"


class Examinator:
    """
    The examinator is responsible for launching exams.
    Examinators take in an index and results of an optimizer run,
    and return a DataFrame of scores for
    the accumulated history performance of the optimizer.
    """

    def __init__(self, exams: Mapping[ExamName, Exam]) -> None:
        self.exams = exams

    def examine(
        self, index: StatefulIndex, results: OrderedDict[int, float]
    ) -> DataFrame:
        scores = {k: v.run(index, results) for k, v in self.exams.items()}
        original = {
            ExamName.STEP_IDX: list(range(len(results))),
            ExamName.ORIGINAL: list(results.values()),
        }
        return DataFrame.from_dict(
            {k.value.lower(): v for k, v in {**original, **scores}.items()}
        )

    @classmethod
    def presets(cls) -> Self:
        """
        Returns:
            The default examinator.
        """

        return cls(
            {
                ExamName.ACC_MIN: Accumulation(AccType.MIN),
                ExamName.ACC_MAX: Accumulation(AccType.MAX),
                ExamName.ACC_AVG: Accumulation(AccType.AVG),
                ExamName.MST_MAX_EDGE_QUERY: MstMaxEdge(MstMaxEdgeType.QUERY),
                ExamName.MST_MAX_EDGE_DATA: MstMaxEdge(MstMaxEdgeType.DATA),
                ExamName.SEGREGATION: Segregation(),
            }
        )
