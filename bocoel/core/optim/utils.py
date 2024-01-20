from typing_extensions import Self


class RemainingSteps:
    def __init__(self, count: int) -> None:
        self._count = count

    @property
    def count(self) -> int:
        return self._count

    def step(self) -> None:
        self._count -= 1

    @property
    def done(self) -> bool:
        # This would never be true if renaming steps if < 0 at first.
        return self._count == 0

    @classmethod
    def infinite(cls) -> Self:
        return cls(count=-1)
