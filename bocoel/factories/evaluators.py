from typing import Any

from bocoel import BigBenchMultipleChoice, BigBenchQuestionAnswer, Evaluator
from bocoel.common import StrEnum


class EvalName(StrEnum):
    BIGBENCH_MC = "BIGBENCH_MULTIPLE_CHOICE"
    BIGBENCH_QA = "BIGBENCH_QUESTION_ANSWER"


def evaluator_factory(name: str | EvalName, /, **kwargs: Any) -> Evaluator:
    name = EvalName.lookup(name)

    match name:
        case EvalName.BIGBENCH_MC:
            return BigBenchMultipleChoice(**kwargs)
        case EvalName.BIGBENCH_QA:
            return BigBenchQuestionAnswer(**kwargs)
