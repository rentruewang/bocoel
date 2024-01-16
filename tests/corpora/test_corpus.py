import pytest

from tests import utils

from . import factories


@pytest.mark.parametrize("device", utils.torch_devices())
def test_init_corpus(device: str) -> None:
    _ = factories.corpus(device)
