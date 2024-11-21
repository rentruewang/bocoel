# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from collections.abc import Sequence

import torch
from numpy.typing import NDArray

from bocoel.models.lms.interfaces import ClassifierModel

from .causal import HuggingfaceCausalLM


class HuggingfaceLogitsLM(HuggingfaceCausalLM, ClassifierModel):
    """
    Logits classification model backed by huggingface's transformers library.
    This means that the model would use the logits of ['1', '2', '3', '4', '5'] as the output,
    if `choices = 5`, for the current batch of inputs.
    """

    def __init__(
        self,
        model_path: str,
        batch_size: int,
        device: str,
        choices: Sequence[str],
        add_sep_token: bool = False,
    ) -> None:
        """
        Parameters:
            model_path: The path to the model.
            batch_size: The batch size to use.
            device: The device to use.
            choices: The choices to classify.
            add_sep_token: Whether to add the sep token.
        """

        super().__init__(
            model_path=model_path,
            batch_size=batch_size,
            device=device,
            add_sep_token=add_sep_token,
        )

        self._choices = choices
        self._encoded_choices = self._encode_tokens(self._choices)

    @property
    def choices(self) -> Sequence[str]:
        return self._choices

    @torch.no_grad()
    def _classify(self, prompts: Sequence[str], /) -> NDArray:
        tokenized = self._tokenizer(prompts)

        output = self._model(**tokenized)

        # Logits has the shape [batch_size, seq_len, vocab_size].
        logits = output.logits

        # Using encoded to select the logits at the last position.
        result = logits[:, -1, self._encoded_choices]

        return result.cpu().numpy()

    def _encode_tokens(self, tokens: Sequence[str]) -> Sequence[int]:
        result: list[int] = []
        for tok in tokens:
            # Only adds the first token because we are only interested in the first token.
            result.append(self._tokenizer.encode(tok, add_special_tokens=False)[0])

        assert len(result) == len(tokens)

        if len(result) != len(set(result)):
            decoded = self._tokenizer.decode(self._tokenizer.encode(tokens))
            raise ValueError(
                "Each token must be converted to 1 unique id."
                f"Got {tokens}, encoded into {decoded}."
            )

        return result
