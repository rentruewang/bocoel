# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
from collections.abc import Sequence
from typing import Protocol

from numpy.typing import NDArray


class ClassifierModel(Protocol):
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.choices})"

    def classify(self, prompts: Sequence[str], /) -> NDArray:
        """
        Classify the given prompts.

        Parameters:
            prompts: The prompts to classify.

        Returns:
            The logits for each prompt and choice.

        Raises:
            ValueError: If the shape of the logits is not [len(prompts), len(choices)].
        """

        classified = self._classify(prompts)

        if list(classified.shape) != [len(prompts), len(self.choices)]:
            raise ValueError(
                f"Expected logits to have shape {[len(prompts), len(self.choices)]}, "
                f"but got {classified.shape}"
            )

        return classified

    @abc.abstractmethod
    def _classify(self, prompts: Sequence[str], /) -> NDArray:
        """
        Generate logits given prompts.

        Parameters:
            prompts: The prompts to classify.

        Returns:
            The logits for each prompt and choice.
        """

        ...

    @property
    @abc.abstractmethod
    def choices(self) -> Sequence[str]:
        """
        The choices for this language model.

        Returns:
            The choices for this language model.
        """

        ...
