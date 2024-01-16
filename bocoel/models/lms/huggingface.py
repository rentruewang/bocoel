from collections.abc import Sequence

import torch
from torch import device
from typing_extensions import Self

from bocoel.models.lms.interfaces import LanguageModel

Device = str | device


class HuggingfaceLM(LanguageModel):
    """
    The Huggingface implementation of LanguageModel.
    This is a wrapper around the Huggingface library,
    which would try to pull the model from the huggingface hub.

    Since huggingface's tokenizer needs padding to the left to work,
    padding doesn't guarentee the same positional embeddings, and thus, results.
    If sameness with generating one by one is desired, batch size should be 1.
    """

    def __init__(
        self, model_path: str, max_len: int, batch_size: int, device: Device
    ) -> None:
        # Optional dependency.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Initializes the tokenizer and pad to the left (this is how it's generated)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(model_path)
        self._model.pad_token = self._tokenizer.pad_token

        self._batch_size = batch_size
        self.max_len = max_len

        self.to(device)

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], /) -> Sequence[str]:
        if not isinstance(prompts, list):
            prompts = list(prompts)

        return sum(
            (
                self.generate_batch(prompts[idx : idx + self.batch])
                for idx in range(0, len(prompts), self.batch)
            ),
            start=[],
        )

    def generate_batch(self, prompt: list[str]) -> list[str]:
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)

        outputs = self._model.generate(**inputs, max_length=self.max_len)
        outputs = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs

    def to(self, device: Device) -> Self:
        self._device = device
        self._model = self._model.to(device)
        return self

    @property
    def device(self) -> Device:
        return self._device

    @property
    def batch(self) -> int:
        return self._batch_size
