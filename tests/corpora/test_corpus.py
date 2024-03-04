import pytest

from bocoel import SbertEmbedder
from tests import utils

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
def test_init_corpus(device: str) -> None:
    embedder = SbertEmbedder(device=device)
    _ = factories.corpus(embedder=embedder)
