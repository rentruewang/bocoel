from bocoel import HuggingfaceLM, LanguageModel
from bocoel.common import StrEnum

from . import common


class LMName(StrEnum):
    HUGGINGFACE = "HUGGINGFACE"


def lm_factory(
    name: str | LMName = LMName.HUGGINGFACE,
    /,
    *,
    model_path: str,
    max_len: int,
    batch_size: int,
    device: str,
) -> LanguageModel:
    if LMName.lookup(name) is not LMName.HUGGINGFACE:
        raise ValueError(f"Unknown corpus name: {name}")

    return common.correct_kwargs(HuggingfaceLM)(
        model_path=model_path, max_len=max_len, batch_size=batch_size, device=device
    )
