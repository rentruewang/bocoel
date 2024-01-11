from collections.abc import Sequence

from .interfaces import Score


class ExactMatch(Score):
    def __call__(self, target: str, references: Sequence[str]) -> float:
        target = self._clean(target)
        references = [self._clean(ref) for ref in references]
        return float(target in references)

    @staticmethod
    def _clean(string: str) -> str:
        return " ".join(string.strip().split())
