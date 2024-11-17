# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from abc import ABCMeta

from .tokenizers import HuggingfaceTokenizer


class HuggingfaceCausalLM(metaclass=ABCMeta):
    """
    The Huggingface implementation of language model.
    This is a wrapper around the Huggingface library,
    which would try to pull the model from the huggingface hub.

    FIXME:
        `add_sep_token` might cause huggingface to bug out with index out of range.
        Still unclear how this might occur as `[SEP]` is a special token.
    """

    def __init__(
        self, model_path: str, batch_size: int, device: str, add_sep_token: bool = False
    ) -> None:
        """
        Parameters:
            model_path: The path to the model.
            batch_size: The batch size to use.
            device: The device to use.
            add_sep_token: Whether to add the sep token.
        """

        # Optional dependency.
        from transformers import AutoModelForCausalLM

        self._model_path = model_path
        self._tokenizer = HuggingfaceTokenizer(
            model_path=model_path, device=device, add_sep_token=add_sep_token
        )

        # Model used for generation
        self._model = AutoModelForCausalLM.from_pretrained(model_path)
        self._model.pad_token = self._tokenizer.pad_token

        self._batch_size = batch_size

        self.to(device)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._model_path})"

    def to(self, device: str, /) -> "HuggingfaceCausalLM":
        self._device = device
        self._tokenizer.to(device)
        self._model = self._model.to(device)
        return self

    @property
    def device(self) -> str:
        return self._device
