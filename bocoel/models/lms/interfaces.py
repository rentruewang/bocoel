import abc
from collections.abc import Sequence
from typing import Protocol

from bocoel.common import Batched


class LanguageModel(Batched, Protocol):
    @abc.abstractmethod
    def generate(self, prompts: Sequence[str], /) -> Sequence[str]:
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

    max_len: int
    """
    Maximum length of the generated text.
    """
