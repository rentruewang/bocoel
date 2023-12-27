from collections.abc import Sequence

from bocoel.models.interfaces import LanguageModel


# TODO: Support VLLM as a backend.
class Vllm(LanguageModel):
    def generate(self, prompt: Sequence[str]) -> Sequence[str]:
        raise NotImplementedError
