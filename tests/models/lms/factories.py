from bocoel import (
    ClassifierModel,
    GenerativeModel,
    HuggingfaceGenerativeLM,
    HuggingfaceLogitsLM,
    HuggingfaceSequenceLM,
)
from tests import utils


@utils.cache
def generative_lm(device: str) -> GenerativeModel:
    return HuggingfaceGenerativeLM(
        model_path="bert-base-uncased",
        device=device,
        batch_size=4,
    )


@utils.cache
def logits_lm(device: str) -> ClassifierModel:
    return HuggingfaceLogitsLM(
        model_path="bert-base-uncased",
        device=device,
        batch_size=4,
        choices=["negative", "positive"],
    )


@utils.cache
def sequence_lm(device: str) -> ClassifierModel:
    return HuggingfaceSequenceLM(
        model_path="bert-base-uncased",
        device=device,
        choices=["negative", "positive"],
    )
