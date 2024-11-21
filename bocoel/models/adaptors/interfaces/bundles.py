# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from numpy.typing import NDArray


class AdaptorBundle(Protocol):
    @abc.abstractmethod
    def evaluate(
        self, data: Mapping[str, Sequence[Any]]
    ) -> Mapping[str, Sequence[float] | NDArray]: ...
