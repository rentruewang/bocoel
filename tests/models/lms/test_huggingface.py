import pytest
from pytest import FixtureRequest

from bocoel import LanguageModel
from tests import utils

from . import factories


@pytest.fixture
def lm_fix(request: FixtureRequest) -> LanguageModel:
    return factories.lm(device=request.param)


@pytest.mark.parametrize("lm_fix", utils.torch_devices(), indirect=True)
def test_lm_generate(lm_fix: LanguageModel) -> None:
    prompts = ["Hello, my name is", "I am a", "I like to eat"]

    gen = lm_fix.generate(prompts)
    assert len(gen) == len(prompts), {
        "gen": gen,
        "prompts": prompts,
    }
    for g, p in zip(gen, prompts):
        assert p in g, {"gen": g, "prompts": p}
