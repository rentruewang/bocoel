from __future__ import annotations

from typing import Protocol, Sequence


class LanguageModel(Protocol):
    def generate(self, prompt: Sequence[str]) -> Sequence[str]:
        ...

    def generate_one(self, prompt: str) -> str:
        return self.generate(prompt=[prompt])[0]
