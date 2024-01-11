from collections.abc import Sequence


class OneHotChoiceAccuracy:
    def __call__(self, target: int, references: Sequence[float]) -> float:
        return references[target]


class MultiChoiceAccuracy:
    def __call__(self, target: int, references: Sequence[int]) -> float:
        return float(target in references)
