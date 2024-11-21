# Copyright (c) 2024 RenChu Wang - All Rights Reserved

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

        Parameters:
            prompts: The prompts to generate.

        Returns:
            The generated responses. The length must be the same as the prompts.

        Todo:
            Add logits.
        """

        ...
