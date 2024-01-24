from collections.abc import Callable

import pytest

from bocoel import HuggingfaceCausalLM
from tests import utils

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
@pytest.mark.parametrize("lm_function", [factories.logits_lm, factories.classifier_lm])
def test_lm_generate(
    device: str, lm_function: Callable[[str], HuggingfaceCausalLM]
) -> None:
    llm = lm_function(device)
    prompts = ["Hello, my name is", "I am a", "I like to eat"]

    gen = llm.generate(prompts)
    assert len(gen) == len(prompts), {
        "gen": gen,
        "prompts": prompts,
    }
    for g, p in zip(gen, prompts):
        assert p.lower() in g.lower(), {"gen": g, "prompts": p}


@pytest.mark.parametrize("device", utils.torch_devices())
@pytest.mark.parametrize("lm_function", [factories.logits_lm, factories.classifier_lm])
def test_lm_classify(
    device: str, lm_function: Callable[[str], HuggingfaceCausalLM]
) -> None:
    llm = lm_function(device)
    prompts = ["Hello, my name is", "I am a", "I like to eat"]

    logits = llm.classify(prompts, choices=2)
    assert len(logits) == len(prompts), {
        "logits": logits,
        "prompts": prompts,
    }
    assert logits.shape[-1] == 2, logits.shape
