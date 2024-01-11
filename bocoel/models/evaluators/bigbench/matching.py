from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from numpy.typing import NDArray

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


class BigBenchMatchType(str, Enum):
    EXACT = "exact"
    NLTK_BLEU = "nltk-bleu"
    SACRE_BLEU = "sacre-bleu"
    ROUGE = "rouge"
    ROUGE_1 = "rouge-score-1"
    ROUGE_2 = "rouge-score-2"
    ROUGE_L = "rouge-score-L"


class BigBenchQuestionAnswer(BigBenchEvalutor):
    def __init__(
        self,
        inputs: str = "inputs",
        targets: str = "targets",
        matching_type: BigBenchMatchType = BigBenchMatchType.EXACT,
    ) -> None:
        self._inputs = inputs
        self._targets = targets

        self._score_fn: Score
        match matching_type:
            case BigBenchMatchType.EXACT:
                self._score_fn = ExactMatch()
            case BigBenchMatchType.NLTK_BLEU:
                self._score_fn = NltkBleuScore()
            case BigBenchMatchType.SACRE_BLEU:
                self._score_fn = SacreBleuScore()
            case BigBenchMatchType.ROUGE:
                self._score_fn = RougeScore()
            case BigBenchMatchType.ROUGE_L:
                self._score_fn = RougeScore2("rougeL")
            case BigBenchMatchType.ROUGE_1:
                self._score_fn = RougeScore2("rouge1")
            case BigBenchMatchType.ROUGE_2:
                self._score_fn = RougeScore2("rouge2")

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
        self, inputs: Sequence[str], targets: Sequence[Sequence[str]], lm: LanguageModel
    ) -> Sequence[float] | NDArray:
        generated = lm.generate(inputs)
        return [self._score_fn(g, t) for g, t in zip(generated, targets)]
