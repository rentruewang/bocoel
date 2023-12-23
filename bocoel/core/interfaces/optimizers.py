from __future__ import annotations

import abc
from typing import Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

from bocoel.corpora import Corpus


class Optimizer(Protocol):
    @abc.abstractmethod
    def optimize(self, corpora: Corpus) -> None:
        ...
