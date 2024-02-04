import abc
from collections.abc import Sequence
from typing import Protocol


class GenerativeModel(Protocol):
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

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
