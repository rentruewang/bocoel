import pytest
from pytest import FixtureRequest

from bocoel import Corpus
from tests import utils

from . import factories


@pytest.fixture
def corpus_fix(request: FixtureRequest) -> Corpus:
    return factories.corpus(device=request.param)


@pytest.mark.parametrize("corpus_fix", utils.torch_devices(), indirect=True)
def test_init_corpus(corpus_fix: Corpus) -> None:
    _ = corpus_fix
