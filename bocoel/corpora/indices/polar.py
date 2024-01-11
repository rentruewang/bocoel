from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.corpora.indices.interfaces import Distance, Index, InternalSearchResult


class PolarIndex(Index):
    """
    Index that uses N-sphere coordinates as interfaces.

    https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    """

    def __init__(
        self,
        embeddings: NDArray,
        distance: str | Distance,
        polar_backend: type[Index],
        **backend_kwargs: Any,
    ) -> None:
        self._index = polar_backend.from_embeddings(
            embeddings=embeddings, distance=Distance(distance), **backend_kwargs
        )

    def _search(self, query: NDArray, k: int = 1) -> InternalSearchResult:
        # Ignores the length of the query. Only direction is preserved.
        spatial = self.polar_to_spatial(1, query)

        return self._index._search(spatial, k=k)

    @property
    def embeddings(self) -> NDArray:
        # Doesn't need to return the polar version of the embeddings
        # because this is just used for looking up encoded embeddings.
        return self._index.embeddings

    @property
    def distance(self) -> Distance:
        return self._index.distance

    @property
    def bounds(self) -> NDArray:
        # See wikipedia linked in the class documentation for details.
        return np.concatenate([[np.pi] * (self.dims - 1), [2 * np.pi]])

    @property
    def dims(self) -> int:
        return self._index.dims - 1

    @classmethod
    def from_embeddings(
        cls, embeddings: NDArray, distance: str | Distance, **kwargs: Any
    ) -> Self:
        return cls(embeddings=embeddings, distance=distance, **kwargs)

    @staticmethod
    def polar_to_spatial(r: float, theta: Sequence[float] | NDArray, /) -> NDArray:
        """
        Convert an N-sphere coordinates to cartesian coordinates.
        See wikipedia linked in the class documentation for details.
        """

        # Add 1 dimension to the front because spherical coordinate's first dimension is r.
        sin = np.concatenate([[1], np.sin(theta)])
        sin = np.cumprod(sin)
        cos = np.concatenate([np.cos(theta), [1]])
        return sin * cos * r

    @staticmethod
    def spatial_to_polar(x: Sequence[float] | NDArray, /) -> tuple[float, NDArray]:
        """
        Convert cartesian coordinates to N-sphere coordinates.
        See wikipedia linked in the class documentation for details.
        """

        # Since the function requires a lot of sum of squares, cache it.
        x_2 = np.array(x[1:]) ** 2

        r: float = np.sqrt(x_2.sum()).item()
        cumsum_back = np.cumsum(x_2[::-1])[::-1]

        theta = np.arctan2(np.sqrt(cumsum_back), x[1:])

        return r, theta
