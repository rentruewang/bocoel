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

    max_len: int
    """
    Maximum length of the generated text.
    """

    batch_size: int
    """
    Batch size for generating text.
    Since huggingface's tokenizer needs padding to the left to work,
    padding doesn't guarentee the same positional embeddings, and thus, results.
    If sameness with generating one by one is desired, batch size should be 1.
    """

    def __init__(
        self, model_path: str, max_len: int, batch_size: int, device: Device
    ) -> None:
        # Initializes the tokenizer and pad to the left (this is how it's generated)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(model_path)
        self._model.pad_token = self._tokenizer.pad_token

        self.max_len = max_len
        self.batch_size = batch_size

        self.to(device)

    def generate(self, prompts: Sequence[str]) -> Sequence[str]:
        if not isinstance(prompts, list):
            prompts = list(prompts)

        return sum(
            (
                self.generate_batch(prompts[idx : idx + self.batch_size])
                for idx in range(0, len(prompts), self.batch_size)
            ),
            start=[],
        )

    def generate_batch(self, prompt: list[str]) -> list[str]:
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)

        outputs = self._model.generate(**inputs, max_length=self.max_len)
        outputs = self._tokenizer.batch_decode(outputs)
        return outputs

    def to(self, device: Device) -> Self:
        self._device = device
        self._model = self._model.to(device)
        return self

    @property
    def device(self) -> Device:
        return self._device
