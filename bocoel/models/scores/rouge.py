from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

from .interfaces import Score


class RougeScore(Score):
    def __init__(self) -> None:
        # Optional dependency.
        from rouge import Rouge

        self._rouge = Rouge()

    def __call__(self, target: str, references: Sequence[str]) -> float:
        return self._rouge.get_scores(target, references)


_RougeScore2Type: TypeAlias = Literal["rouge1", "rouge2", "rougeL"]


class RougeScore2(Score):
    def __init__(self, typ: _RougeScore2Type) -> None:
        from rouge_score.rouge_scorer import RougeScorer

        self._scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self._typ = typ

    def __call__(self, target: Any, references: Sequence[Any]) -> Any:
        if len(references) >= 1:
            raise ValueError

        return self._scorer.score(target=references[0], prediction=target)[self._typ]
