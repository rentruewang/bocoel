from typing import Sequence

from bocoel.models.interfaces import LanguageModel


# TODO: Support VLLM as a backend.
class VllmLM(LanguageModel):
    def generate(self, prompt: Sequence[str]) -> Sequence[str]:
        raise NotImplementedError
