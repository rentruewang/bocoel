from bocoel import HuggingfaceLogitsLM, LanguageModel
from tests import utils


@utils.cache
def lm(device: str) -> LanguageModel:
    return HuggingfaceLogitsLM(model_path="distilgpt2", device=device, batch_size=4)
