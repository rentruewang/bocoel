# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from collections.abc import Sequence

import torch
from numpy.typing import NDArray

from bocoel.models.lms.interfaces import ClassifierModel

from .tokenizers import HuggingfaceTokenizer


class HuggingfaceSequenceLM(ClassifierModel):
    """
    The sequence classification model backed by huggingface's transformers library.
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        choices: Sequence[str],
        add_sep_token: bool = False,
    ) -> None:
        # Optional dependency
        from transformers import AutoModelForSequenceClassification

        self._model_path = model_path
        self._tokenizer = HuggingfaceTokenizer(
            model_path=model_path, device=device, add_sep_token=add_sep_token
        )

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

    def to(self, device: str, /) -> "HuggingfaceSequenceLM":
        self._tokenizer.to(device)
        self._classifier.to(device)
        return self
