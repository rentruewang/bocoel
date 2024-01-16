from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bocoel.corpora.embedders.interfaces import Embedder

_OUTPUT_KEY = "type_vocab_size"


class HuggingfaceEmbedder(Embedder):
    def __init__(
        self,
        path: str,
        device: str = "cpu",
        batch_size: int = 64,
        transform: Callable[[Any], Tensor] = lambda output: output.logits,
    ) -> None:
        self._model = AutoModelForSequenceClassification.from_pretrained(path)
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._batch_size = batch_size

        self._device = device
        self._model = self._model.to(device)
        self._transform = transform

    @property
    def batch(self) -> int:
        return self._batch_size

    @property
    def dims(self) -> int:
        return self._model.config[_OUTPUT_KEY]

    @torch.no_grad()
    def _encode(self, texts: Sequence[str], /) -> NDArray:
        results = []
        for idx in range(0, len(texts), self.batch):
            batch = texts[idx : idx + self.batch]
            encoded = self._encode_batch(batch)
            results.append(encoded)
        return np.concatenate(results, axis=0)

    @torch.no_grad()
    def _encode_batch(self, texts: Sequence[str]) -> NDArray:
        encoded = self._tokenizer(texts).to(self._device)
        output = self._model(**encoded)
        return self._transform(output).cpu().numpy()
