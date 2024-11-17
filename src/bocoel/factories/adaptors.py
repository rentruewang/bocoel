# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from typing import Any

from bocoel import (
    Adaptor,
    BigBenchMultipleChoice,
    BigBenchQuestionAnswer,
    GlueAdaptor,
    Sst2QuestionAnswer,
)
from bocoel.common import StrEnum

from . import common


class AdaptorName(StrEnum):
    """
    The names of the adaptors.
    """

    BIGBENCH_MC = "BIGBENCH_MULTIPLE_CHOICE"
    "Corresponds to `BigBenchMultipleChoice`."

    BIGBENCH_QA = "BIGBENCH_QUESTION_ANSWER"
    "Corresponds to `BigBenchQuestionAnswer`."

    SST2 = "SST2"
    "Corresponds to `Sst2QuestionAnswer`."

    GLUE = "GLUE"
    "Corresponds to `GlueAdaptor`."


def adaptor(name: str | AdaptorName, /, **kwargs: Any) -> Adaptor:
    """
    Create an adaptor.

    Parameters:
        name: The name of the adaptor.
        **kwargs: The keyword arguments to pass to the adaptor.
            See the documentation of the corresponding adaptor for details.

    Returns:
        The adaptor instance.

    Raises:
        ValueError: If the name is unknown.
    """

    name = AdaptorName.lookup(name)

    match name:
        case AdaptorName.BIGBENCH_MC:
            return common.correct_kwargs(BigBenchMultipleChoice)(**kwargs)
        case AdaptorName.BIGBENCH_QA:
            return common.correct_kwargs(BigBenchQuestionAnswer)(**kwargs)
        case AdaptorName.SST2:
            return common.correct_kwargs(Sst2QuestionAnswer)(**kwargs)
        case AdaptorName.GLUE:
            return common.correct_kwargs(GlueAdaptor)(**kwargs)
        case _:
            raise ValueError(f"Unknown adaptor name: {name}")
