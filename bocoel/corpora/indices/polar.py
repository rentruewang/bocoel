from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bocoel.corpora.indices import utils
from bocoel.corpora.indices.interfaces import Boundary, Distance, Index, InternalResult


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
        embeddings = utils.normalize(embeddings)
        self._index = polar_backend(
            embeddings=embeddings,
            distance=distance,
            **backend_kwargs,
        )

    def _search(self, query: NDArray, k: int = 1) -> InternalResult:
        # Ignores the length of the query. Only direction is preserved.
        spatial = self.polar_to_spatial(np.ones([len(query)]), query)

        assert spatial.shape[1] == self.dims + 1

        return self._index._search(spatial, k=k)

    @property
    def batch(self) -> int:
        return self._index.batch

    @property
    def data(self) -> NDArray:
        """
        Returns the actual data of the polar index.

        Doesn't need to return the polar version of the embeddings
        because this is just used for looking up encoded embeddings.
        """

        return self._index.data

    @property
    def distance(self) -> Distance:
        return self._index.distance

    @property
    def boundary(self) -> Boundary:
        # See wikipedia linked in the class documentation for details.
        upper = np.concatenate([[np.pi] * (self.dims - 1), [2 * np.pi]])
        lower = np.zeros_like(upper)
        return Boundary(np.stack([lower, upper], axis=-1))

    @property
    def dims(self) -> int:
        return self._index.dims - 1

    @staticmethod
    def polar_to_spatial(r: ArrayLike, theta: ArrayLike) -> NDArray:
        """
        Convert an N-sphere coordinates to cartesian coordinates.
        See wikipedia linked in the class documentation for details.
        """

        r = np.array(r)
        theta = np.array(theta)

        if r.ndim != 1:
            raise ValueError(f"Expected r to be 1D, got {r.ndim}")

        if theta.ndim != 2:
            raise ValueError(f"Expected theta to be 2D, got {theta.ndim}")

        if r.shape[0] != theta.shape[0]:
            raise ValueError(
                f"Expected r and theta to have the same length, got {r.shape[0]} and {theta.shape[0]}"
            )

        # Add 1 dimension to the front because spherical coordinate's first dimension is r.
        sin = np.concatenate([np.ones([len(r), 1]), np.sin(theta)], axis=1)
        sin = np.cumprod(sin, axis=1)
        cos = np.concatenate([np.cos(theta), np.ones([len(r), 1])], axis=1)
        return sin * cos * r[:, None]

    @staticmethod
    def spatial_to_polar(x: ArrayLike) -> tuple[NDArray, NDArray]:
        """
        Convert cartesian coordinates to N-sphere coordinates.
        See wikipedia linked in the class documentation for details.
        """

        x = np.array(x)

        if x.ndim != 2:
            raise ValueError(f"Expected x to be 2D, got {x.ndim}")

        # Since the function requires a lot of sum of squares, cache it.
        x_2 = x[:, 1:] ** 2

        r = np.sqrt(x_2.sum(axis=1))
        cumsum_back = np.cumsum(x_2[:, ::-1], axis=1)[:, ::-1]

        theta = np.arctan2(np.sqrt(cumsum_back), x[:, 1:])
        return r, theta
