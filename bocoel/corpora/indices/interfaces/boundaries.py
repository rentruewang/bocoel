import dataclasses as dcls

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self


@dcls.dataclass(frozen=True)
class Boundary:
    bounds: NDArray

    def __post_init__(self) -> None:
        if self.bounds.ndim != 2:
            raise ValueError(f"Expected 2D bounds, got {self.bounds.ndim}D")

        if self.bounds.shape[1] != 2:
            raise ValueError(f"Expected 2 columns, got {self.bounds.shape[1]}")

        if (self.lower > self.upper).any():
            raise ValueError("Expected lower <= upper")

    def __len__(self) -> int:
        return self.dims

    def __getitem__(self, idx: int, /) -> NDArray:
        return self.bounds[idx]

    @property
    def dims(self) -> int:
        return self.bounds.shape[0]

    @property
    def lower(self) -> NDArray:
        return self.bounds[:, 0]

    @property
    def upper(self) -> NDArray:
        return self.bounds[:, 1]

    @classmethod
    def fixed(cls, lower: float, upper: float, dims: int) -> Self:
        return cls(bounds=np.array([[lower, upper]] * dims))
