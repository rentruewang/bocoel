from __future__ import annotations

from typing import Protocol, Sequence


# FIXME: Should I set generate to take in a sequence of strings or just a string?
class LanguageModel(Protocol):
    def generate(self, prompt: Sequence[str]) -> Sequence[str]:
        ...

    def generate_one(self, prompt: str) -> str:
        return self.generate(prompt=[prompt])[0]
