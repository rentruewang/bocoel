from collections.abc import Sequence

import torch
from numpy.typing import NDArray
from transformers import AutoModelForSequenceClassification

from bocoel.models.lms.huggingface.bases import Device

from .bases import HuggingfaceBaseLM


class HuggingfaceClassifierLM(HuggingfaceBaseLM):
    def __init__(
        self, model_path: str, batch_size: int, device: Device, choices: Sequence[str]
    ) -> None:
        super().__init__(model_path, batch_size, device)

        classifier = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._classifier = classifier.to(device)
        self._classifier.config.pad_token_id = self._tokenizer.pad_token_id

        self._choices = choices

    @torch.no_grad()
    def _classify(self, prompts: Sequence[str], /, choices: Sequence[str]) -> NDArray:
        if tuple(choices) != tuple(self._choices):
            raise ValueError(f"choices must be {self._choices}. Got {choices}.")

        tokenized = self._tokenize(prompts)
        output = self._classifier(**tokenized)

        return output.logits.cpu().numpy()
