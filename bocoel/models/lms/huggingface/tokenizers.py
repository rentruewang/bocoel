from collections.abc import Sequence

from typing_extensions import Self


class HuggingfaceTokenizer:
    def __init__(self, model_path: str, device: str) -> None:
        # Optional dependency.
        from transformers import AutoTokenizer

        # Initializes the tokenizer and pad to the left for sequence generation.
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left", truncation_side="left"
        )
        if (eos := self._tokenizer.eos_token) is not None:
            self._tokenizer.pad_token = eos
        else:
            self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        if self._tokenizer.sep_token is None:
            self._tokenizer.add_special_tokens({"sep_token": "[SEP]"})

        self._device = device

    def to(self, device: str, /) -> Self:
        self._device = device
        return self

    def __call__(self, prompts: Sequence[str], /):
        return self.tokenize(prompts)

    def tokenize(self, prompts: Sequence[str], /):
        """
        Tokenize, pad, truncate, cast to device, and yield the encoded results.
        Returning `BatchEncoding` but not marked in the type hint
        due to optional dependency.
        """
        if not isinstance(prompts, list):
            prompts = list(prompts)

        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            max_length=self._tokenizer.model_max_length,
            padding=True,
            truncation=True,
        )
        return inputs.to(self.device)

    def encode(
        self,
        prompts: Sequence[str],
        /,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ):
        return self._tokenizer.encode(
            prompts,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
        )

    def decode(self, outputs, /, skip_special_tokens: bool = True):
        return self._tokenizer.decode(outputs, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, outputs, /, skip_special_tokens: bool = True):
        return self._tokenizer.batch_decode(
            outputs, skip_special_tokens=skip_special_tokens
        )

    @property
    def device(self) -> str:
        return self._device

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    @property
    def pad_token(self) -> str:
        return self._tokenizer.pad_token
