from collections.abc import Sequence

from torch import Tensor, device
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing_extensions import Self

from bocoel.models.interfaces import LanguageModel

Device = str | device


class HuggingfaceLM(LanguageModel):
    """
    The Huggingface implementation of LanguageModel.
    This is a wrapper around the Huggingface library,
    which would try to pull the model from the huggingface hub.
    """

    def __init__(self, model_path: str, max_len: int, device: Device) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, pad_token_id=self._tokenizer.eos_token_id
        )
        self._max_len = max_len
        self.to(device)

        # TODO: Verify that this works.
        # Source: https://stackoverflow.com/a/73137031
        if self._tokenizer.pad_token is None:
            self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self._model.resize_token_embeddings(len(self._tokenizer))

    def generate(self, prompt: Sequence[str]) -> Sequence[str]:
        # FIXME: Perhaps there is a better way to perform tokenization?
        inputs = self._tokenizer(
            prompt, return_tensors="pt", padding="max_length", max_length=self._max_len
        )
        inputs = inputs.to(self.device)
        outputs = self._model.generate(
            **inputs, max_length=self._max_len, num_return_sequences=1
        )
        outputs = self._tokenizer.batch_decode(outputs)
        return outputs

    def to(self, device: Device) -> Self:
        self._device = device
        self._model = self._model.to(device)
        return self

    @property
    def device(self) -> Device:
        return self._device
