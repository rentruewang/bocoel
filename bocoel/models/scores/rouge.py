from collections.abc import Sequence

from .interfaces import Score


class RougeScore(Score):
    def __init__(self) -> None:
        # Optional dependency.
        from rouge import Rouge

        self._rouge = Rouge()

    def __call__(self, target: str, references: Sequence[str]) -> float:
        return self._rouge.get_scores(target, references)
