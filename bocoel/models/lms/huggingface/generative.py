from collections.abc import Sequence

import torch
from numpy.typing import NDArray

from bocoel.models.lms.huggingface.causal import Device

from .causal import HuggingfaceCausalLM


class HuggingfaceLogitsLM(HuggingfaceCausalLM):
    def __init__(self, model_path: str, batch_size: int, device: Device) -> None:
        super().__init__(model_path, batch_size, device)

    @torch.no_grad()
    def _classify(self, prompts: Sequence[str], /, choices: int) -> NDArray:
        tokenized = self._tokenize(prompts)

        # Encode tokens and select the logits at the given output.
        tokens = [str(i) for i in range(1, choices + 1)]
        encoded = self._encode_tokens(tokens)

        output = self._model(**tokenized)

        # Logits has the shape [batch_size, seq_len, vocab_size].
        logits = output.logits

        # Using encoded to select the logits at the last position.
        result = logits[:, -1, encoded]

        return result.cpu().numpy()

    def _encode_tokens(self, tokens: Sequence[str]) -> Sequence[int]:
        result: list[int] = sum(
            [self._tokenizer.encode(tok, add_special_tokens=False) for tok in tokens],
            [],
        )
        if len(result) != len(tokens):
            raise ValueError(f"Tokens must be words. Got {tokens}.")
        return result
