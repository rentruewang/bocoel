from collections.abc import Sequence

from torch import device
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
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(model_path)
        self._max_len = max_len
        self.to(device)

    def generate(self, prompts: Sequence[str]) -> Sequence[str]:
        # FIXME: Only able to generate one by one for now, due to padding issues.
        return [self._generate_one(prompt=p) for p in prompts]

    def _generate_one(self, prompt: str) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)

        outputs = self._model.generate(**inputs, max_length=self._max_len)
        outputs = self._tokenizer.batch_decode(outputs)[0]
        return outputs

    def to(self, device: Device) -> Self:
        self._device = device
        self._model = self._model.to(device)
        return self

    @property
    def device(self) -> Device:
        return self._device
