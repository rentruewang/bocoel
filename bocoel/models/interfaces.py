from typing import Protocol


class LanguageModel(Protocol):
    def generate(self, prompt: str) -> str:
        ...
