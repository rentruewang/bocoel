# Copyright (c) BoCoEL Authors - All Rights Reserved

from collections.abc import Mapping, Sequence
from typing import Any

from numpy.typing import NDArray

from bocoel.adaptors.interfaces import Adaptor, AdaptorBundle

__all__ = ["AdaptorMapping"]


class AdaptorMapping(AdaptorBundle):
    def __init__(self, adaptors: Mapping[str, Adaptor]) -> None:
        self._adaptors = adaptors

    def evaluate(
        self, data: Mapping[str, Sequence[Any]]
    ) -> Mapping[str, Sequence[float] | NDArray]:
        return {name: ev.evaluate(data) for name, ev in self._adaptors.items()}
