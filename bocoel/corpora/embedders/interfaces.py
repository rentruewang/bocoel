import abc
from collections.abc import Sequence
from typing import Protocol

import structlog
from numpy.typing import NDArray

from bocoel.common import Batched

LOGGER = structlog.get_logger()


class Embedder(Batched, Protocol):
    """
    Embedders are responsible for encoding text into vectors.
    Embedders in this project are considered volatile because it requires CPU time,
    unless some database that encodes this functionality is found.
    """

    def encode(self, text: Sequence[str], /) -> NDArray:
        """
        Calls the encode function and performs some checks.
        """

        encoded = self._encode(text)

        if (dim := encoded.shape[-1]) != self.dims:
            raise ValueError(
                f"Expected the encoded embeddings to have dimension {self.dims}, got {dim}"
            )

        return encoded

    @property
    @abc.abstractmethod
    def dims(self) -> int:
        """
        The dimensions of the embeddings
        """

        ...

    @abc.abstractmethod
    def _encode(self, texts: Sequence[str], /) -> NDArray:
        """
        Implements the encode function.

        Parameters
        ----------

        `text: Sequence[str]`
        The text to encode.
        If a string is given, it is treated as a singleton batch.
        If a list is given, all those embeddings are processed together in a single batch.

        Returns
        -------

        A numpy array of shape [batch, dims]. If the input is a string, the shape would be [dims].
        """

        ...
