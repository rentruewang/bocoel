from bocoel import HuggingfaceClassifierLM, HuggingfaceLogitsLM, LanguageModel
from tests import utils


@utils.cache
def logits_lm(device: str) -> LanguageModel:
    return HuggingfaceLogitsLM(
        model_path="bert-base-uncased", device=device, batch_size=4
    )


@utils.cache
def classifier_lm(device: str) -> LanguageModel:
    return HuggingfaceClassifierLM(
        model_path="bert-base-uncased", device=device, batch_size=4
    )
