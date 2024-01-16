import pytest

from tests import utils

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
def test_lm_generate(device: str) -> None:
    lm = factories.lm(device)
    prompts = ["Hello, my name is", "I am a", "I like to eat"]

    gen = lm.generate(prompts)
    assert len(gen) == len(prompts), {
        "gen": gen,
        "prompts": prompts,
    }
    for g, p in zip(gen, prompts):
        assert p in g, {"gen": g, "prompts": p}
