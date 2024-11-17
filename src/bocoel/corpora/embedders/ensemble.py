# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import os
from collections.abc import Sequence

import torch
from torch import Tensor

from bocoel.corpora.embedders.interfaces import Embedder


class EnsembleEmbedder(Embedder):
    """
    An ensemble of embedders. The embeddings are concatenated together.
    """

    def __init__(self, embedders: Sequence[Embedder], sequential: bool = False) -> None:
        """
        Parameters:
            embedders: The embedders to use.
            sequential: Whether to use sequential processing.

        Raises:
            ValueError: If the embedders have different batch sizes.
        """

        # Check if all embedders have the same batch size.
        self._embedders = embedders
        self._batch_size = embedders[0].batch
        if len(set(emb.batch for emb in embedders)) != 1:
            raise ValueError("All embedders must have the same batch size")

        self._sequential = sequential

        cpus = os.cpu_count()
        assert cpus is not None
        self._cpus = cpus

    def __repr__(self) -> str:
        return f"Ensemble({[str(emb) for emb in self._embedders]})"

    @property
    def batch(self) -> int:
        return self._batch_size

    @property
    def dims(self) -> int:
        return sum(emb.dims for emb in self._embedders)

    def _encode(self, texts: Sequence[str]) -> Tensor:
        results = [emb._encode(texts) for emb in self._embedders]
        return torch.cat([res.cpu() for res in results], dim=-1)
