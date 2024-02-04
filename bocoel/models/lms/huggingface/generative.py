from collections.abc import Sequence

import torch
from typing_extensions import Self

from bocoel.models.lms.interfaces import GenerativeModel

from .tokenizers import HuggingfaceTokenizer


class HuggingfaceGenerativeLM(GenerativeModel):
    """
    The Huggingface implementation of LanguageModel.
    This is a wrapper around the Huggingface library,
    which would try to pull the model from the huggingface hub.

    Since huggingface's tokenizer needs padding to the left to work,
    padding doesn't guarentee the same positional embeddings, and thus, results.
    If sameness with generating one by one is desired, batch size should be 1.
    """

    def __init__(self, model_path: str, batch_size: int, device: str) -> None:
        # Optional dependency.
        from transformers import AutoModelForCausalLM

        self._model_path = model_path
        self._tokenizer = HuggingfaceTokenizer(model_path=model_path, device=device)

        # Model used for generation
        self._model = AutoModelForCausalLM.from_pretrained(model_path)
        self._model.pad_token = self._tokenizer.pad_token

        self._batch_size = batch_size

        self.to(device)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._model_path})"

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], /) -> Sequence[str]:
        results: list[str] = []
        for idx in range(0, len(prompts), self._batch_size):
            results.extend(self._generate_batch(prompts[idx : idx + self._batch_size]))
        return results

    def to(self, device: str, /) -> Self:
        self._device = device
        self._tokenizer.to(device)
        self._model = self._model.to(device)
        return self

    def _generate_batch(self, prompts: Sequence[str]) -> list[str]:
        inputs = self._tokenizer.tokenize(prompts)
        outputs = self._model.generate(**inputs)
        outputs = self._tokenizer.batch_decode(outputs)
        return outputs

    @property
    def device(self) -> str:
        return self._device
