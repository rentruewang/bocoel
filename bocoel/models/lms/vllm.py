from collections.abc import Sequence

from bocoel.models.lms.interfaces import LanguageModel


# TODO: Support VLLM as a backend.
class Vllm(LanguageModel):
    def generate(self, prompts: Sequence[str]) -> Sequence[str]:
        raise NotImplementedError
