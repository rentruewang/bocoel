import abc
from collections.abc import Sequence
from typing import Protocol

from numpy.typing import NDArray


class LanguageModel(Protocol):
    @abc.abstractmethod
    def generate(self, prompts: Sequence[str], /) -> Sequence[str]:
        """
        Generate a sequence of responses given prompts.
        The length of the response is the same as the prompt.
        The response would be a continuation of the prompt,
        and the prompts would be the prefix of the response.

        Parameters
        ----------

        `prompts: Sequence[str]`
        The prompts to generate responses from.


        Returns
        -------

        A sequence of responses.
        This has the same length as the prompt.
        Each response is a string.
        """

        ...

    def classify(self, prompts: Sequence[str], /, choices: Sequence[str]) -> NDArray:
        """
        Generate logits given prompts.

        Parameters
        ----------

        `prompts: Sequence[str]`
        The prompts to generate logits from.

        `choices: Sequence[str]`
        The choices for this batch of prompts.

        Returns
        -------

        A list of logits
        Each logit has the same length given by each prompt's choices.
        """

        classified = self._classify(prompts, choices=choices)

        if classified.shape != (len(prompts), len(choices)):
            raise ValueError(
                f"Expected logits to have shape {(len(prompts), len(choices))}, "
                f"but got {classified.shape}"
            )

        return classified

    @abc.abstractmethod
    def _classify(self, prompts: Sequence[str], /, choices: Sequence[str]) -> NDArray:
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
