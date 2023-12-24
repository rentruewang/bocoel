from __future__ import annotations

import abc
from typing import Protocol, Sequence


# FIXME: Should I set generate to take in a sequence of strings or just a string?
class LanguageModel(Protocol):
    @abc.abstractmethod
    def generate(self, prompt: Sequence[str]) -> Sequence[str]:
        ...
