from collections.abc import Sequence

import torch
from numpy.typing import NDArray

from bocoel.models.lms.huggingface.bases import Device

from .bases import HuggingfaceBaseLM


class HuggingfaceLogitsLM(HuggingfaceBaseLM):
    """
    The Huggingface implementation of LanguageModel that uses logits in classification.
    This means that the model would use the logits of ['1', '2', '3', '4', '5'] as the output,
    if `choices = 5`, for the current batch of inputs.
    """

    def __init__(self, model_path: str, batch_size: int, device: Device) -> None:
        super().__init__(model_path, batch_size, device)

    @torch.no_grad()
    def _classify(self, prompts: Sequence[str], /, choices: Sequence[str]) -> NDArray:
        tokenized = self._tokenize(prompts)

        # Encode tokens and select the logits at the given output.
        encoded = self._encode_tokens(choices)

        output = self._model(**tokenized)

        # Logits has the shape [batch_size, seq_len, vocab_size].
        logits = output.logits

        # Using encoded to select the logits at the last position.
        result = logits[:, -1, encoded]

        return result.cpu().numpy()

    def _encode_tokens(self, tokens: Sequence[str]) -> Sequence[int]:
        result: list[int] = []
        for tok in tokens:
            result.extend(self._tokenizer.encode(tok, add_special_tokens=False))

        if len(result) != len(tokens):
            decoded = self._tokenizer.decode(self._tokenizer.encode(tokens))
            raise ValueError(
                "Tokens must be words. Each token must be converted to 1 id."
                f"Got {tokens}, encoded into {decoded}."
            )

        return result
