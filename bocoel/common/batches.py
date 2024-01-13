import abc
from typing import Protocol


class Batched(Protocol):
    @property
    @abc.abstractmethod
    def batch(self) -> int:
        """
        The batch size used for processing.
        """

        ...
