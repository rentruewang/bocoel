from collections.abc import Sequence

import torch
from numpy.typing import NDArray
from transformers import AutoModelForSequenceClassification

from bocoel.models.lms.interfaces import ClassifierModel

from .tokenizers import HuggingfaceTokenizer


class HuggingfaceSequenceLM(ClassifierModel):
    def __init__(
        self,
        model_path: str,
        device: str,
        choices: Sequence[str],
    ) -> None:
        self._model_path = model_path
        self._tokenizer = HuggingfaceTokenizer(model_path=model_path, device=device)

        self._choices = choices

        classifier = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._classifier = classifier.to(device)
        self._classifier.config.pad_token_id = self._tokenizer.pad_token_id

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._model_path}, {self._choices})"

    @property
    def choices(self) -> Sequence[str]:
        return self._choices

    @torch.no_grad()
    def _classify(self, prompts: Sequence[str], /) -> NDArray:
        tokenized = self._tokenizer(prompts)
        output = self._classifier(**tokenized)
        return output.logits.cpu().numpy()
