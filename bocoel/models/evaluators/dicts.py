from collections.abc import Mapping, Sequence
from typing import Any

from numpy.typing import NDArray

from bocoel.models.evaluators.interfaces import Evaluator, EvaluatorBundle
from bocoel.models.lms import LanguageModel


class EvaluatorDict(EvaluatorBundle):
    def __init__(self, evaluators: Mapping[str, Evaluator]) -> None:
        self._evals = evaluators

    def evaluate(
        self, data: Mapping[str, Sequence[Any]], lm: LanguageModel
    ) -> Mapping[str, Sequence[float] | NDArray]:
        return {name: ev.evaluate(data, lm) for name, ev in self._evals.items()}
