from collections.abc import Sequence

from .interfaces import Score


class OneHotChoiceAccuracy(Score):
    def __call__(self, target: int, references: Sequence[float]) -> float:
        return references[target]


class MultiChoiceAccuracy(Score):
    def __call__(self, target: int, references: Sequence[int]) -> float:
        return float(target in references)
