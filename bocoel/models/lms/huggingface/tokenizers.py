# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from collections.abc import Sequence
from typing import Any

from transformers import BatchEncoding


class HuggingfaceTokenizer:
    """
    A tokenizer for Huggingface models.
    """

    def __init__(self, model_path: str, device: str, add_sep_token: bool) -> None:
        """
        Parameters:
            model_path: The path to the model.
            device: The device to use.
            add_sep_token: Whether to add the sep token.

        Raises:
            ImportError: If the transformers library is not installed.
        """

        # Optional dependency.
        from transformers import AutoTokenizer

        # Initializes the tokenizer and pad to the left for sequence generation.
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left", truncation_side="left"
        )

        # Always add the pad token.
        if (eos := self._tokenizer.eos_token) is not None:
            self._tokenizer.pad_token = eos
        else:
            self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        if add_sep_token:
            if self._tokenizer.sep_token is None:
                self._tokenizer.add_special_tokens({"sep_token": "[SEP]"})

        self._device = device

    def to(self, device: str, /) -> "HuggingfaceTokenizer":
        """
        Move the tokenizer to the given device.

        Parameters:
            device: The device to move to.
        """
        self._device = device
        return self

    def tokenize(
        self, prompts: Sequence[str], /, max_length: int | None = None
    ) -> BatchEncoding:
        """
        Tokenize, pad, truncate, cast to device, and yield the encoded results.
        Returning `BatchEncoding` but not marked in the type hint
        due to optional dependency.

        Parameters:
            prompts: The prompts to tokenize.

        Returns:
            (BatchEncoding): The tokenized prompts.
        """
        if not isinstance(prompts, list):
            prompts = list(prompts)

        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return inputs.to(self.device)

    def __call__(
        self, prompts: Sequence[str], /, max_length: int | None = None
    ) -> BatchEncoding:
        return self.tokenize(prompts, max_length=max_length)

    def encode(
        self,
        prompts: Sequence[str],
        /,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """
        Encode the given prompts.

        Parameters:
            prompts: The prompts to encode.
            return_tensors: Whether to return tensors.
            add_special_tokens: Whether to add special tokens.

        Returns:
            (Any): The encoded prompts.
        """

        return self._tokenizer.encode(
            prompts,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
        )

    def decode(self, outputs: Any, /, skip_special_tokens: bool = True) -> str:
        """
        Decode the given outputs.

        Parameters:
            outputs: The outputs to decode.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            The decoded outputs.
        """

        return self._tokenizer.decode(outputs, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self, outputs: Any, /, skip_special_tokens: bool = True
    ) -> list[str]:
        """
        Batch decode the given outputs.

        Parameters:
            outputs: The outputs to decode.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            The batch decoded outputs.
        """

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
