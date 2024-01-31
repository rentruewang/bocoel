import abc
from typing import Any, Protocol


class Score(Protocol):
    @abc.abstractmethod
    def __call__(self, target: Any, references: list[Any]) -> float:
        """
        Calculate the score of a target given references.

        Parameters
        ----------

        `target: Any`
        The target to calculate the score for.

        `references: list[Any]`
        The references to calculate the score from.

        Returns
        -------

        The score of the target given the references.
        """

        ...
