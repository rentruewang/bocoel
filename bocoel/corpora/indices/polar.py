# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from typing import Any

import numpy as np
import structlog
from numpy.typing import ArrayLike, NDArray

from bocoel.corpora.indices import utils
from bocoel.corpora.indices.interfaces import Boundary, Distance, Index, InternalResult

LOGGER = structlog.get_logger()


class PolarIndex(Index):
    """
    Index that uses N-sphere coordinates as interfaces.
    See wikipedia linked below for details.

    Converting the spatial indices into spherical coordinates has the following benefits:

    - Since the coordinates are normalized, the radius is always 1.
    - The search region is rectangular in spherical coordinates,
        ideal for bayesian optimization.

    [Wikipedia link on N-sphere](https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates)
    """

    def __init__(
        self,
        embeddings: NDArray,
        distance: str | Distance,
        *,
        polar_backend: type[Index],
        **backend_kwargs: Any,
    ) -> None:
        """
        Parameters:
            embeddings: The embeddings to index.
            distance: The distance metric to use.
            polar_backend: The backend to use for indexing.
            **backend_kwargs: The backend specific keyword arguments.
        """

        embeddings = utils.normalize(embeddings)
        self._index = polar_backend(
            embeddings=embeddings,
            distance=distance,
            **backend_kwargs,
        )

        dims = self._index.dims - 1

        self._boundary = self._polar_boundary(dims)
        self._data = self._polar_coordinates()

    def _search(self, query: NDArray, k: int = 1) -> InternalResult:
        # Ignores the length of the query. Only direction is preserved.
        spatial = self.polar_to_spatial(np.ones([len(query)]), query)

        assert spatial.shape[1] == self.dims + 1, (
            "Spatial dimensions do not match embeddings. "
            f"Expected {self.dims + 1}. Got {spatial.shape[1]}."
        )

        return self._index._search(spatial, k=k)

    @property
    def batch(self) -> int:
        return self._index.batch

    @property
    def data(self) -> NDArray:
        return self._data

    @property
    def distance(self) -> Distance:
        return self._index.distance

    @property
    def boundary(self) -> Boundary:
        return self._boundary

    def _polar_boundary(self, dims: int) -> Boundary:
        """
        The boundary of the queries.
        For polar coordinate it is [0, pi] for all dimensions
        except the last one which is [0, 2 * pi].

        Returns:
            The boundary of the input.
        """

        # See wikipedia linked in the class documentation for details.
        upper = np.concatenate([[np.pi] * (dims - 1), [2 * np.pi]])
        lower = np.zeros_like(upper)
        return Boundary(np.stack([lower, upper], axis=-1))

    def _polar_coordinates(self) -> NDArray:
        LOGGER.info(
            "Converting embeddings to polar coordinates.", batch_size=self.batch
        )

        embeddings = self._index.data

        results = []
        for idx in range(0, len(embeddings), self.batch):
            batch = embeddings[idx : idx + self.batch]
            _, polar = self.spatial_to_polar(batch)
            results.append(polar)

        transformed = np.concatenate(results, axis=0)
        assert (
            transformed.shape[1] == self._index.dims - 1
        ), "Polar dimensions do not match embeddings."

        return transformed

    @staticmethod
    def polar_to_spatial(r: ArrayLike, theta: ArrayLike) -> NDArray:
        """
        Convert an N-sphere coordinates to cartesian coordinates.
        See wikipedia linked in the class documentation for details.

        Parameters:
            r: The radius of the N-sphere. Has the shape [N].
            theta: The angles of the N-sphere. Hash the shape [N, D].

        Returns:
            The cartesian coordinates of the N-sphere.
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

        Parameters:
            x: The cartesian coordinates. Has the shape [N, D].

        Returns:
            A tuple. The radius and the angles of the N-sphere.
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
