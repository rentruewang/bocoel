from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any

import structlog
from numpy.typing import NDArray

from bocoel.common import StrEnum
from bocoel.models.adaptors import utils
from bocoel.models.lms import LanguageModel
from bocoel.models.scores import MultiChoiceAccuracy, OneHotChoiceAccuracy, Score

from . import prompts
from .interfaces import BigBenchAdaptor

LOGGER = structlog.get_logger()


class BigBenchChoiceType(StrEnum):
    SUM_OF_SCORES = "SUM_OF_SCORES"
    LIST_OF_ANSWERS = "LIST_OF_ANSWERS"

    @property
    def score(self) -> Score:
        match self:
            case BigBenchChoiceType.SUM_OF_SCORES:
                return OneHotChoiceAccuracy()
            case BigBenchChoiceType.LIST_OF_ANSWERS:
                return MultiChoiceAccuracy()


class BigBenchMultipleChoice(BigBenchAdaptor):
    def __init__(
        self,
        inputs: str = "inputs",
        multiple_choice_targets: str = "multiple_choice_targets",
        multiple_choice_scores: str = "multiple_choice_scores",
        choice_type: str | BigBenchChoiceType = BigBenchChoiceType.SUM_OF_SCORES,
    ) -> None:
        self._inputs = inputs
        self._multiple_choice_targets = multiple_choice_targets
        self._multiple_choice_scores = multiple_choice_scores

        self._score_fn = BigBenchChoiceType.lookup(choice_type).score

    def evaluate(
        self, data: Mapping[str, Any], lm: LanguageModel
    ) -> Sequence[float] | NDArray:
        # Get data.
        inputs = data[self._inputs]
        multiple_choice_targets = data[self._multiple_choice_targets]
        multiple_choice_scores = data[self._multiple_choice_scores]

        LOGGER.debug(
            "Evaluating",
            inputs=inputs,
            multiple_choice_targets=multiple_choice_targets,
            multiple_choice_scores=multiple_choice_scores,
        )

        # Check data.
        if not all(isinstance(ipt, str) for ipt in inputs):
            raise ValueError("Inputs must be strings.")

        if not all(utils.list_of(mct, str) for mct in multiple_choice_targets):
            raise ValueError("Multiple choice targets must be sequences.")

        if not all(utils.list_of(mcs, Number) for mcs in multiple_choice_scores):
            raise ValueError("Multiple choice scores must be floats.")

        return self._evaluate(
            inputs=inputs,
            multiple_choice_targets=multiple_choice_targets,
            multiple_choice_scores=multiple_choice_scores,
            lm=lm,
        )

    def _evaluate(
        self,
        inputs: Sequence[str],
        multiple_choice_targets: Sequence[Sequence[str]],
        multiple_choice_scores: Sequence[Sequence[float]],
        lm: LanguageModel,
    ) -> Sequence[float] | NDArray:
        prmpt = [
            prompts.numeric_choices(question=q, choices=c)
            for q, c in zip(inputs, multiple_choice_targets)
        ]

        generated = lm.generate(prmpt)

        gen_int = [utils.parse_int(item) for item in generated]

        LOGGER.debug("Generated prompts", generated=generated, gen_int=gen_int)

        return [
            self._score_fn(target=g, references=s)
            for g, s in zip(gen_int, multiple_choice_scores)
        ]
