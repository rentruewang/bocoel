import dataclasses as dcls

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self


@dcls.dataclass(frozen=True)
class Boundary:
    """
    The boundary of embeddings in a corpus.
    The boundary is defined as a hyperrectangle in the embedding space.
    """

    bounds: NDArray
    """
    The boundary arrays of the corpus.
    Must be of shape `[dims, 2]`, where dims is the number of dimensions.
    The first column is the lower bound, the second column is the upper bound.
    """

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
        "The number of dimensions."
        return self.bounds.shape[0]

    @property
    def lower(self) -> NDArray:
        "The lower bounds. Must be of shape `[dims]`."
        return self.bounds[:, 0]

    @property
    def upper(self) -> NDArray:
        "The upper bounds. Must be of shape `[dims]`."
        return self.bounds[:, 1]

    @classmethod
    def fixed(cls, lower: float, upper: float, dims: int) -> Self:
        """
        Create a boundary with fixed bounds.
        If `lower > upper`, a `ValueError` would be raised.

        Parameters
        ----------

        `lower: float`
        The lower bound.

        `upper: float`
        The upper bound.

        `dims: int`
        The number of dimensions.

        Returns
        -------

        A `Boundary` instance.
        """

        if lower > upper:
            raise ValueError("Expected lower <= upper")

        return cls(bounds=np.array([[lower, upper]] * dims))
