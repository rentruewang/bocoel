import functools

from bocoel import HuggingfaceLM, LanguageModel


@functools.cache
def lm(device: str) -> LanguageModel:
    return HuggingfaceLM(
        model_path="distilgpt2", device=device, batch_size=4, max_len=512
    )
