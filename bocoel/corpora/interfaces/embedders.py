from __future__ import annotations

import abc
from typing import Protocol, Sequence

from numpy.typing import NDArray


class Embedder(Protocol):
    """
    Embedders are responsible for encoding text into vectors.
    Embedders in this project are considered volatile because it requires CPU time,
    unless some database that encodes this functionality is found.
    """

    dims: int
    """
    The dimensions of the embeddings
    """

    def __call__(self, text: str | Sequence[str]) -> NDArray:
        """
        Calls the encode function and performs some checks.
        """

        encoded = self.encode(text)

        if (dim := encoded.shape[-1]) != self.dims:
            raise ValueError(
                f"Expected the encoded embeddings to have dimension {self.dims}, got {dim}"
            )

        return encoded

    @abc.abstractmethod
    def encode(self, text: str | Sequence[str]) -> NDArray:
        """
        Implements the encode function.

        Parameters
        ----------

        `text: str | Sequence[str]`
        The text to encode.
        If a string is given, it is treated as a singleton batch.
        If a list is given, all those embeddings are processed together in a single batch.

        Returns
        -------

        A numpy array of shape [batch, dims]. If the input is a string, the shape would be [dims].
        """

        ...
