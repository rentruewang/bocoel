from bocoel import HuggingfaceClassifierLM, HuggingfaceLogitsLM, LanguageModel
from bocoel.common import StrEnum

from . import common


class LMName(StrEnum):
    LOGITS = "LOGITS"
    CLASSIFIER = "CLASSIFIER"


def lm_factory(
    name: str | LMName = LMName.LOGITS,
    /,
    *,
    model_path: str,
    batch_size: int,
    device: str,
) -> LanguageModel:
    match LMName.lookup(name):
        case LMName.LOGITS:
            return common.correct_kwargs(HuggingfaceLogitsLM)(
                model_path=model_path, batch_size=batch_size, device=device
            )
        case LMName.CLASSIFIER:
            return common.correct_kwargs(HuggingfaceClassifierLM)(
                model_path=model_path, batch_size=batch_size, device=device
            )
        case _:
            raise ValueError(f"Unknown LM name {name}")
