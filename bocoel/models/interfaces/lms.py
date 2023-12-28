import abc
from collections.abc import Sequence
from typing import Protocol


# FIXME: Should I set generate to take in a sequence of strings or just a string?
class LanguageModel(Protocol):
    @abc.abstractmethod
    def generate(self, prompts: Sequence[str]) -> Sequence[str]:
        """
        Generate a sequence of responses given prompts.

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
