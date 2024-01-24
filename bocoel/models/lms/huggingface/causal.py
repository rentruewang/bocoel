from abc import ABCMeta
from collections.abc import Sequence

import torch
from torch import device
from typing_extensions import Self

from bocoel.models.lms.interfaces import LanguageModel

Device = str | device


class HuggingfaceCausalLM(LanguageModel, metaclass=ABCMeta):
    """
    The Huggingface implementation of LanguageModel.
    This is a wrapper around the Huggingface library,
    which would try to pull the model from the huggingface hub.

    Since huggingface's tokenizer needs padding to the left to work,
    padding doesn't guarentee the same positional embeddings, and thus, results.
    If sameness with generating one by one is desired, batch size should be 1.
    """

    def __init__(self, model_path: str, batch_size: int, device: Device) -> None:
        # Optional dependency.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Initializes the tokenizer and pad to the left for sequence generation.
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left", truncation_side="left"
        )
        if (eos := self._tokenizer.eos_token) is not None:
            self._tokenizer.pad_token = eos
        else:
            self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Model used for generation
        self._model = AutoModelForCausalLM.from_pretrained(model_path)
        self._model.pad_token = self._tokenizer.pad_token

        self._batch_size = batch_size

        self.to(device)

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], /) -> Sequence[str]:
        results: list[str] = []
        for idx in range(0, len(prompts), self._batch_size):
            results.extend(self._generate_batch(prompts[idx : idx + self._batch_size]))
        return results

    def to(self, device: Device) -> Self:
        self._device = device
        self._model = self._model.to(device)
        return self

    def _tokenize(self, prompts: Sequence[str], /):
        if not isinstance(prompts, list):
            prompts = list(prompts)

        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            max_length=self._tokenizer.model_max_length,
            padding=True,
            truncation=True,
        )
        return inputs.to(self.device)

    def _generate_batch(self, prompts: Sequence[str]) -> list[str]:
        inputs = self._tokenize(prompts)
        outputs = self._model.generate(**inputs)
        outputs = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs

    @property
    def device(self) -> Device:
        return self._device
