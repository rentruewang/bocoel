from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any

import structlog
import typeguard
from numpy.typing import NDArray

from bocoel.common import StrEnum
from bocoel.models.lms import ClassifierModel
from bocoel.models.scores import MultiChoiceAccuracy, OneHotChoiceAccuracy, Score

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
        lm: ClassifierModel,
        inputs: str = "inputs",
        multiple_choice_targets: str = "multiple_choice_targets",
        multiple_choice_scores: str = "multiple_choice_scores",
        choice_type: str | BigBenchChoiceType = BigBenchChoiceType.SUM_OF_SCORES,
    ) -> None:
        self.lm = lm

        self.inputs = inputs
        self.multiple_choice_targets = multiple_choice_targets
        self.multiple_choice_scores = multiple_choice_scores

        self._score_fn = BigBenchChoiceType.lookup(choice_type).score

    def evaluate(self, data: Mapping[str, Any]) -> Sequence[float] | NDArray:
        # Get data.
        inputs = data[self.inputs]
        multiple_choice_targets = data[self.multiple_choice_targets]
        multiple_choice_scores = data[self.multiple_choice_scores]

        LOGGER.debug(
            "Evaluating",
            inputs=inputs,
            multiple_choice_targets=multiple_choice_targets,
            multiple_choice_scores=multiple_choice_scores,
        )

        # Check data.
        typeguard.check_type("inputs", inputs, Sequence[str])
        typeguard.check_type("mct", multiple_choice_targets, Sequence[Sequence[str]])
        typeguard.check_type("mcs", multiple_choice_scores, Sequence[Sequence[Number]])

        prompts = [
            self.numeric_choices(question=q, choices=c)
            for q, c in zip(inputs, multiple_choice_targets)
        ]

        # Get the maximum number of choices.
        # Usually every question should have the same number of choices (5).
        num_choices_per_question = [len(mcs) for mcs in multiple_choice_scores]

        if min(num_choices_per_question) == 0:
            raise ValueError(
                "Multiple choice scores must not be empty. "
                f"Got {multiple_choice_scores}"
            )

        choices = [str(i) for i in range(1, max(num_choices_per_question) + 1)]
        if any(choice not in self.lm.choices for choice in choices):
            raise ValueError(
                f"Choices {choices} are not in the language model's choices {self.lm.choices}"
            )

        # Apply classification on the prompts.
        selected = self.lm.classify(prompts)

        # Chosen has shape [batch_size].
        # Although choices start from 1, chosen is the index of the choice.
        chosen = selected.argmax(axis=-1)

        LOGGER.debug("Generated prompts", chosen=chosen)

        return [
            self._score_fn(target=g, references=s)
            for g, s in zip(chosen, multiple_choice_scores)
        ]

    @staticmethod
    def numeric_choices(question: str, choices: Sequence[str]) -> str:
        """
        Convert a multiple choice question into a numeric choice question.
        Returns a tuple of generated prompt and list of valid choices.
        """

        return (
            f"{question}\nSelect from one of the following (answer in number):\n"
            + "\n".join(f"{i}) {choice}" for i, choice in enumerate(choices, 1))
        )
