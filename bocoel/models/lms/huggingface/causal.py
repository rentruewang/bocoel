from abc import ABCMeta

from typing_extensions import Self

from .tokenizers import HuggingfaceTokenizer


class HuggingfaceCausalLM(metaclass=ABCMeta):
    """
    The Huggingface implementation of LanguageModel.
    This is a wrapper around the Huggingface library,
    which would try to pull the model from the huggingface hub.
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

    def to(self, device: str, /) -> Self:
        self._device = device
        self._tokenizer.to(device)
        self._model = self._model.to(device)
        return self

    @property
    def device(self) -> str:
        return self._device
