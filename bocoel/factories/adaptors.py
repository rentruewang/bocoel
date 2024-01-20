from typing import Any

from bocoel import Adaptor, BigBenchMultipleChoice, BigBenchQuestionAnswer
from bocoel.common import StrEnum

from . import common


class AdaptorName(StrEnum):
    BIGBENCH_MC = "BIGBENCH_MULTIPLE_CHOICE"
    BIGBENCH_QA = "BIGBENCH_QUESTION_ANSWER"


def adaptor_factory(name: str | AdaptorName, /, **kwargs: Any) -> Adaptor:
    name = AdaptorName.lookup(name)

    match name:
        case AdaptorName.BIGBENCH_MC:
            return common.correct_kwargs(BigBenchMultipleChoice)(**kwargs)
        case AdaptorName.BIGBENCH_QA:
            return common.correct_kwargs(BigBenchQuestionAnswer)(**kwargs)
