from bocoel import (
    ClassifierModel,
    GenerativeModel,
    HuggingfaceLogitsLM,
    HuggingfaceSequenceLM,
)
from tests import utils


@utils.cache
def logits_lm(device: str) -> GenerativeModel:
    return HuggingfaceLogitsLM(
        model_path="bert-base-uncased",
        device=device,
        batch_size=4,
        choices=["negative", "positive"],
    )


@utils.cache
def classifier_lm(device: str) -> ClassifierModel:
    return HuggingfaceSequenceLM(
        model_path="bert-base-uncased",
        device=device,
        choices=["negative", "positive"],
    )
