from collections.abc import Sequence


class RougeScore:
    def __init__(self) -> None:
        # Optional dependency.
        from rouge import Rouge

        self._rouge = Rouge()

    def __call__(self, target: str, references: Sequence[str]) -> float:
        return self._rouge.get_scores(target, references)
