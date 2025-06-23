# Copyright (c) BoCoEL Authors - All Rights Reserved

from typing import Any

from bocoel import (
    Adaptor,
    BigBenchMultipleChoice,
    BigBenchQuestionAnswer,
    GlueAdaptor,
    Sst2QuestionAnswer,
)

from . import common

__all__ = ["adaptor"]


@common.correct_kwargs
def adaptor(name: str, /, **kwargs: Any) -> Adaptor:
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

    match name:
        case "BIGBENCH_MC":
            return BigBenchMultipleChoice(**kwargs)
        case "BIGBENCH_QA":
            return BigBenchQuestionAnswer(**kwargs)
        case "SST2":
            return Sst2QuestionAnswer(**kwargs)
        case "GLUE":
            return GlueAdaptor(**kwargs)
        case _:
            raise ValueError(f"Unknown adaptor name: {name}")
