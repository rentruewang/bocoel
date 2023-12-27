import numpy as np
import pytest
from numpy.typing import NDArray

from bocoel import Index, WhiteningIndex

from . import test_hnswlib


def whiten_index(embeddings: NDArray) -> Index:
    return WhiteningIndex(embeddings=embeddings, dist="cosine", remains=3)


def test_init_whiten_index() -> None:
    embeddings = test_hnswlib.emb()
    idx = whiten_index(embeddings)
    assert idx.dims == 3
