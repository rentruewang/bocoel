import abc
import typing
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, Protocol

from numpy.typing import NDArray

from bocoel.models.evaluators import utils
from bocoel.models.evaluators.interfaces import Evaluator
from bocoel.models.lms import LanguageModel
from bocoel.models.scores import MultiChoiceAccuracy, OneHotChoiceAccuracy

from . import prompts


@typing.runtime_checkable
class MultiChoiceScore(Protocol):
    @abc.abstractmethod
    def __call__(self, target: Any, references: Sequence[Any]) -> float:
        ...


class MultipleChoiceType(str, Enum):
    ONE_HOT = "one-hot"
    MULTIPLE_CHOICE = "multi-choice"


class BigBenchMultipleChoice(Evaluator):
    def __init__(
        self,
        inputs: str = "inputs",
        multiple_choice_targets: str = "multiple_choice_targets",
        multiple_choice_scores: str = "multiple_choice_scores",
        choice_type: MultipleChoiceType = MultipleChoiceType.ONE_HOT,
    ) -> None:
        self._inputs = inputs
        self._multiple_choice_targets = multiple_choice_targets
        self._multiple_choice_scores = multiple_choice_scores

        self._score_fn: MultiChoiceScore
        match choice_type:
            case MultipleChoiceType.ONE_HOT:
                self._score_fn = OneHotChoiceAccuracy()
            case MultipleChoiceType.MULTIPLE_CHOICE:
                self._score_fn = MultiChoiceAccuracy()
        assert isinstance(self._score_fn, MultiChoiceScore)

    def evaluate(
        self, collated: Mapping[str, Any], lm: LanguageModel
    ) -> Sequence[float] | NDArray:
        # Get data.
        inputs = collated[self._inputs]
        multiple_choice_targets = collated[self._multiple_choice_targets]
        multiple_choice_scores = collated[self._multiple_choice_scores]

        # Check data.
        if not all(isinstance(ipt, str) for ipt in inputs):
            raise ValueError("Inputs must be strings.")

        if not all(utils.list_of(mct, int) for mct in multiple_choice_targets):
            raise ValueError("Multiple choice targets must be sequences.")

        if not all(utils.list_of(mcs, float) for mcs in multiple_choice_scores):
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

        return [
            self._score_fn(target=g, references=s)
            for g, s in zip(gen_int, multiple_choice_scores)
        ]
