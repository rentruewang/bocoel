from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bocoel.corpora.indices import utils
from bocoel.corpora.indices.interfaces import (
    Boundary,
    Distance,
    Index,
    IndexedArray,
    InternalResult,
)
from bocoel.corpora.indices.utils import Indexer


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
        spatial = batched_polar_to_spatial([1.0] * len(query), query)

        return self._index._search(spatial, k=k)

    @property
    def batch(self) -> int:
        return self._index.batch

    @property
    def _embeddings(self) -> NDArray | IndexedArray:
        # Doesn't need to return the polar version of the embeddings
        # because this is just used for looking up encoded embeddings.
        return Indexer(self._index._embeddings, mapping=batched_spatial_to_angle)

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


def polar_to_spatial(r: float, theta: Sequence[float] | NDArray, /) -> NDArray:
    """
    Convert an N-sphere coordinates to cartesian coordinates.
    See wikipedia linked in the class documentation for details.
    """

    theta = np.array(theta)

    # Add 1 dimension to the front because spherical coordinate's first dimension is r.
    sin = np.concatenate([[1], np.sin(theta)])
    sin = np.cumprod(sin)
    cos = np.concatenate([np.cos(theta), [1]])
    return sin * cos * r


def batched_polar_to_spatial(
    r: Sequence[float] | NDArray, theta: Sequence[Sequence[float]] | NDArray, /
) -> NDArray:
    return np.array(
        [polar_to_spatial(radius, angle) for radius, angle in zip(r, theta)]
    )


# TODO: Adapt this to support batches for more efficient computation.
def spatial_to_polar(x: Sequence[float] | NDArray, /) -> tuple[float, NDArray]:
    """
    Convert cartesian coordinates to N-sphere coordinates.
    See wikipedia linked in the class documentation for details.
    """

    x = np.array(x)

    # Since the function requires a lot of sum of squares, cache it.
    x_2 = np.array(x[1:]) ** 2

    r: float = np.sqrt(x_2.sum()).item()
    cumsum_back = np.cumsum(x_2[::-1])[::-1]

    theta = np.arctan2(np.sqrt(cumsum_back), x[1:])

    return r, theta


def batched_spatial_to_polar(
    x: Sequence[Sequence[float]] | NDArray, /
) -> tuple[NDArray, NDArray]:
    rs = []
    thetas = []
    for point in x:
        r, theta = spatial_to_polar(point)
        rs.append(r)
        thetas.append(theta)
    return np.array(rs), np.array(thetas)


def batched_spatial_to_angle(x: Sequence[Sequence[float]] | NDArray, /) -> NDArray:
    r, theta = batched_spatial_to_polar(x)

    # FIXME: Somehow r != 1
    # if not np.allclose(r, 1):
    #     raise ValueError(f"Expected all r to be 1, got {r}")

    return theta
