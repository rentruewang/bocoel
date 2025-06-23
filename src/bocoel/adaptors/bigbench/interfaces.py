# Copyright (c) BoCoEL Authors - All Rights Reserved

from typing import Protocol

from bocoel.adaptors.interfaces import Adaptor
from bocoel.scores import Score

__all__ = ["BigBenchAdaptor"]


class BigBenchAdaptor(Adaptor, Protocol):
    _score_fn: Score
