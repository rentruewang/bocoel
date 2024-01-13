from enum import Enum

from bocoel import HuggingfaceLM, LanguageModel


class LMName(str, Enum):
    HUGGINGFACE = "huggingface"


def lm_factory(
    name: str | LMName = LMName.HUGGINGFACE,
    /,
    *,
    model_path: str,
    max_len: int,
    batch_size: int,
    device: str,
) -> LanguageModel:
    if name is not LMName.HUGGINGFACE:
        raise ValueError(f"Unknown corpus name: {name}")

    return HuggingfaceLM(
        model_path=model_path, max_len=max_len, batch_size=batch_size, device=device
    )
