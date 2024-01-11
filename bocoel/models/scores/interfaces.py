import abc
from collections.abc import Sequence
from typing import Any, Protocol


class Score(Protocol):
    @abc.abstractmethod
    def __call__(self, target: Any, references: Sequence[Any]) -> Any:
        ...
