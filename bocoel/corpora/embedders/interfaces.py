import abc
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from torch import Tensor
from tqdm import tqdm

from bocoel import common
from bocoel.corpora.storages import Storage

LOGGER = structlog.get_logger()


class Embedder(Protocol):
    """
    Embedders are responsible for encoding text into vectors.
    Embedders in this project are considered volatile because it requires CPU time,
    unless some database that encodes this functionality is found.
    """

    def __repr__(self) -> str:
        name = common.remove_base_suffix(self, Embedder)
        return f"{name}({self.dims})"

    def encode_storage(
        self,
        storage: Storage,
        /,
        transform: Callable[[Mapping[str, Sequence[Any]]], Sequence[str]],
    ) -> NDArray:
        results: list[NDArray] = []

        for idx in tqdm(range(0, len(storage), self.batch)):
            LOGGER.debug(
                "Encoding storage",
                storage=storage,
                batch_size=self.batch,
                idx=idx,
                total=len(storage),
            )
            batch = storage[idx : idx + self.batch]
            texts = transform(batch)
            encoded = self.encode(texts)
            results.append(encoded)

        return np.concatenate(results, axis=0)

    def encode(self, text: Sequence[str], /) -> NDArray:
        """
        Calls the encode function and performs some checks.
        """

        with torch.no_grad():
            encoded = self._encode(text)

        if (dim := encoded.shape[-1]) != self.dims:
            raise ValueError(
                f"Expected the encoded embeddings to have dimension {self.dims}, got {dim}"
            )

        return encoded.cpu().numpy()

    @property
    @abc.abstractmethod
    def batch(self) -> int:
        """
        The batch size to use when encoding.
        """

        ...

    @property
    @abc.abstractmethod
    def dims(self) -> int:
        """
        The dimensions of the embeddings
        """

        ...

    @abc.abstractmethod
    def _encode(self, texts: Sequence[str], /) -> Tensor:
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

        A tensor of shape [batch, dims]. If the input is a string, the shape would be [dims].
        """

        ...
