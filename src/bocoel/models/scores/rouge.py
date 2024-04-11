from typing import Any, Literal

import typeguard

from .interfaces import Score

_RougeScoreType = Literal["rouge-1", "rouge-2", "rouge-l"]


class RougeScore(Score):
    def __init__(self, metric: _RougeScoreType) -> None:
        # Optional dependency.
        from rouge import Rouge

        self._rouge = Rouge()
        self._metric = metric

    def __call__(self, target: str, references: list[str]) -> float:
        typeguard.check_type("references", references, list[str])

        if len(references) != 1:
            raise ValueError(
                f"References must be a sequence of length 1. Got: {references}"
            )

        scores = self._rouge.get_scores(target, references[0])

        if len(scores) != 1:
            raise ValueError(
                f"References must be a sequence of length 1. Got: {references}"
            )
        return scores[0][self._metric]["f"]


_RougeScore2Type = Literal["rouge1", "rouge2", "rougeL"]


class RougeScore2(Score):
    def __init__(self, typ: _RougeScore2Type) -> None:
        from rouge_score.rouge_scorer import RougeScorer

        self._scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self._typ = typ

    def __call__(self, target: Any, references: list[str]) -> float:
        typeguard.check_type("references", references, list[str])

        if len(references) != 1:
            raise ValueError(
                f"References must be a sequence of length 1. Got: {references}"
            )

        return self._scorer.score(target=references[0], prediction=target)[
            self._typ
        ].fmeasure
