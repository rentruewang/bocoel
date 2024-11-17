# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import typeguard

from .interfaces import Score


class OneHotChoiceAccuracy(Score):
    def __call__(self, target: int, references: list[float]) -> float:
        typeguard.check_type(references, list[float])
        return references[target]


class MultiChoiceAccuracy(Score):
    def __call__(self, target: int, references: list[int]) -> float:
        typeguard.check_type(references, list[int])
        return float(target in references)
