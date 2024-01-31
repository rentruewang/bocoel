from collections.abc import Sequence

from bocoel import ClassifierModel, HuggingfaceLogitsLM, HuggingfaceSequenceLM
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
    choices: Sequence[str],
) -> ClassifierModel:
    match LMName.lookup(name):
        case LMName.LOGITS:
            return common.correct_kwargs(HuggingfaceLogitsLM)(
                model_path=model_path,
                batch_size=batch_size,
                device=device,
                choices=choices,
            )
        case LMName.CLASSIFIER:
            return common.correct_kwargs(HuggingfaceSequenceLM)(
                model_path=model_path, device=device, choices=choices
            )
        case _:
            raise ValueError(f"Unknown LM name {name}")
