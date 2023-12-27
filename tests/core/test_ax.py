import pytest
from pytest import FixtureRequest

from bocoel import AxServiceCore, Core
from tests import utils
from tests.corpora import test_corpus
from tests.models.evaluators import test_bleu
from tests.models.lms import test_huggingface


def core(device: str) -> Core:
    corpus = test_corpus.corpus(device=device)
    lm = test_huggingface.lm(device=device)
    evaluator = test_bleu.bleu()
    return AxServiceCore(corpus=corpus, lm=lm, evaluator=evaluator)


@pytest.fixture
def core_fix(request: FixtureRequest) -> Core:
    return core(request.param)


@pytest.mark.parametrize("core_fix", utils.devices(), indirect=True)
def test_init_core(core_fix: Core) -> None:
    ...


@pytest.mark.parametrize("core_fix", utils.devices(), indirect=True)
def test_optimize(core_fix: Core) -> None:
    # FIXME: Sobol has 5 iterations.
    # This should be changed when the hardcode is removed.
    for _ in range(5):
        state = core_fix.optimize()
