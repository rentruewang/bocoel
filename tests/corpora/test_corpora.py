import pytest
from pytest import FixtureRequest

from bocoel import ComposedCorpus, Corpus, DataFrameStorage, SBertEmbedder
from tests import utils

from .indices import test_hnswlib
from .storages import test_df_storage


def corpus(device: str) -> Corpus:
    storage = DataFrameStorage(test_df_storage.df())
    embedder = SBertEmbedder(device=device)
    index = test_hnswlib.index(embedder.encode(storage.get("question")))

    return ComposedCorpus(index=index, embedder=embedder, storage=storage)


@pytest.fixture
def corpus_fix(request: FixtureRequest) -> Corpus:
    return corpus(device=request.param)


@pytest.mark.parametrize("corpus_fix", utils.devices(), indirect=True)
def test_init_corpus(corpus_fix: Corpus) -> None:
    ...
