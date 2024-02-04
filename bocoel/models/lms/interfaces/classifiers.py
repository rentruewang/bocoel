import abc
from collections.abc import Sequence
from typing import Protocol

from numpy.typing import NDArray


class ClassifierModel(Protocol):
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.choices})"

    def classify(self, prompts: Sequence[str], /) -> NDArray:
        """
        Generate logits given prompts.

        Parameters
        ----------

        `prompts: Sequence[str]`
        The prompts to generate logits from.

        Returns
        -------

        A list of logits
        Each logit has the same length given by each prompt's choices.
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

        Parameters
        ----------

        `prompts: Sequence[str]`
        The prompts to generate logits from.

        `choices: Sequence[str]`
        Number of choices for this batch of prompts.

        Returns
        -------

        A list of logits. Must have the shaep [batch_size, choices].
        """

        ...

    @property
    def choices(self) -> Sequence[str]:
        """
        The choices for this language model.
        """

        ...
