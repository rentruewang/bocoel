import pytest
from pytest import FixtureRequest

from bocoel import HuggingfaceLM, LanguageModel
from tests import utils


def lm(device: str) -> LanguageModel:
    return HuggingfaceLM(model_path="distilgpt2", device=device, max_len=512)


@pytest.fixture
def lm_fix(request: FixtureRequest) -> LanguageModel:
    return lm(device=request.param)


@pytest.mark.parametrize("lm_fix", utils.devices(), indirect=True)
def test_lm_generate(lm_fix: LanguageModel) -> None:
    prompts = ["Hello, my name is", "I am a", "I like to eat"]

    gen: list[str] = lm_fix.generate(prompts)
    assert len(gen) == len(prompts), {
        "gen": gen,
        "prompts": prompts,
    }
    for g, p in zip(gen, prompts):
        assert p == g[: len(p)], {"gen": g, "prompts": p}
