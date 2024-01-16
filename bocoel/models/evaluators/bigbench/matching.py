from collections.abc import Mapping, Sequence
from typing import Any

from numpy.typing import NDArray

from bocoel.common import StrEnum
from bocoel.models.evaluators import utils
from bocoel.models.lms import LanguageModel
from bocoel.models.scores import (
    ExactMatch,
    NltkBleuScore,
    RougeScore,
    RougeScore2,
    SacreBleuScore,
    Score,
)

from .interfaces import BigBenchEvalutor


class BigBenchMatchType(StrEnum):
    EXACT = "EXACT"
    NLTK_BLEU = "NLTK_BLEU"
    SACRE_BLEU = "SACRE_BLEU"
    ROUGE_1 = "ROUGE_1"
    ROUGE_2 = "ROUGE_2"
    ROUGE_L = "ROUGE_L"
    ROUGE_SCORE_1 = "ROUGE_SCORE_1"
    ROUGE_SCORE_2 = "ROUGE_SCORE_2"
    ROUGE_SCORE_L = "ROUGE_SCORE_L"

    @property
    def score(self) -> Score:
        match self:
            case BigBenchMatchType.EXACT:
                return ExactMatch()
            case BigBenchMatchType.NLTK_BLEU:
                return NltkBleuScore()
            case BigBenchMatchType.SACRE_BLEU:
                return SacreBleuScore()
            case BigBenchMatchType.ROUGE_L:
                return RougeScore("rouge-l")
            case BigBenchMatchType.ROUGE_1:
                return RougeScore("rouge-1")
            case BigBenchMatchType.ROUGE_2:
                return RougeScore("rouge-2")
            case BigBenchMatchType.ROUGE_SCORE_L:
                return RougeScore2("rougeL")
            case BigBenchMatchType.ROUGE_SCORE_1:
                return RougeScore2("rouge1")
            case BigBenchMatchType.ROUGE_SCORE_2:
                return RougeScore2("rouge2")


class BigBenchQuestionAnswer(BigBenchEvalutor):
    def __init__(
        self,
        inputs: str = "inputs",
        targets: str = "targets",
        matching_type: str | BigBenchMatchType = BigBenchMatchType.EXACT,
    ) -> None:
        self._inputs = inputs
        self._targets = targets

        self._score_fn = BigBenchMatchType.lookup(matching_type).score

    def evaluate(
        self, data: Mapping[str, Sequence[Any]], lm: LanguageModel
    ) -> Sequence[float] | NDArray:
        # Get data.
        inputs = data[self._inputs]
        targets = data[self._targets]

        # Check data.
        if not all(isinstance(ipt, str) for ipt in inputs):
            raise ValueError("Inputs must be strings.")

        if not all(utils.list_of(tgt, str) for tgt in targets):
            raise ValueError("Targets must be strings.")

        return self._evaluate(inputs, targets, lm)

    def _evaluate(
        self, inputs: Sequence[str], targets: Sequence[str], lm: LanguageModel
    ) -> Sequence[float] | NDArray:
        generated = lm.generate(inputs)
        return [self._score_fn(g, [t]) for g, t in zip(generated, targets)]
