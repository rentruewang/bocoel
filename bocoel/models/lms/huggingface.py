from typing import Sequence

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
        self._model = AutoModelForCausalLM.from_pretrained(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._max_len = max_len
        self.to(device)

    def generate(self, prompt: Sequence[str]) -> Sequence[str]:
        input_ids: Tensor = self._tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        outputs = self._model.generate(
            input_ids, max_length=self._max_len, num_return_sequences=1
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
