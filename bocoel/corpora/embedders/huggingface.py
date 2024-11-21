# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from collections.abc import Callable, Sequence
from typing import Any

from torch import Tensor

from bocoel.corpora.embedders.interfaces import Embedder


class HuggingfaceEmbedder(Embedder):
    """
    Huggingface embedder. Uses the transformers library.
    Not a traditional encoder but uses a classifier and logits as embeddings.
    """

    def __init__(
        self,
        path: str,
        device: str = "cpu",
        batch_size: int = 64,
        transform: Callable[[Any], Tensor] = lambda output: output.logits,
    ) -> None:
        """
        Initializes the Huggingface embedder.

        Parameters:
            path: The path to the model.
            device: The device to use.
            batch_size: The batch size for encoding.
            transform: The transformation function to use.

        Raises:
            ImportError: If transformers is not installed.
            ValueError: If the model does not have a `config.id2label` attribute.
        """

        # Optional dependency.
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._path = path
        self._model = AutoModelForSequenceClassification.from_pretrained(path)
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._batch_size = batch_size

        self._device = device
        self._model = self._model.to(device)
        self._transform = transform

        try:
            self._dims = len(self._model.config.id2label)
        except AttributeError as e:
            raise ValueError(
                "The model must have a `config.id2label` attribute to determine the number of classes."
            ) from e

    def __repr__(self) -> str:
        return f"Huggingface({self._path}, {self.dims})"

    @property
    def batch(self) -> int:
        return self._batch_size

    @property
    def dims(self) -> int:
        return self._dims

    def _encode(self, texts: Sequence[str]) -> Tensor:
        encoded = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            max_length=self._tokenizer.model_max_length,
        ).to(self._device)
        output = self._model(**encoded)

        transformed = self._transform(output)
        return transformed
