import abc
from collections.abc import Sequence
from typing import Protocol

from numpy.typing import NDArray


class LanguageModel(Protocol):
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

    @abc.abstractmethod
    def logits(self, prompts: Sequence[str], /) -> NDArray:
        """
        Generate logits given prompts.

        Parameters
        ----------

        `prompts: Sequence[str]`
        The prompts to generate responses from.

        Returns
        -------

        A batch of logits.
        This has the shape [batch, sequence length, num_tokens].
        """

        ...

    @abc.abstractmethod
    def encode_tokens(self, tokens: Sequence[str], /) -> Sequence[int]:
        """
        Encode tokens into integers.

        Parameters
        ----------

        `tokens: Sequence[str]`
        The tokens to encode.
        Every token must be a word and only correspond to an integer.

        Returns
        -------

        A sequence of integers.
        """

        ...
