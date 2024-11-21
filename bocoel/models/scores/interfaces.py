# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
from typing import Any, Protocol


class Score(Protocol):
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    @abc.abstractmethod
    def __call__(self, target: Any, references: list[Any]) -> float:
        """
        Evaluate the target with respect to the references.

        Parameters:
            target: The target to evaluate.
            references: The references to evaluate against.

        Returns:
            The score for the target.
        """

        ...
