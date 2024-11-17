# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import typeguard

from .interfaces import Score


class ExactMatch(Score):
    def __call__(self, target: str, references: list[str]) -> float:
        typeguard.check_type(references, list[str])

        target = self._clean(target)
        references = [self._clean(ref) for ref in references]
        return float(target in references)

    @staticmethod
    def _clean(string: str) -> str:
        return " ".join(string.strip().split())
