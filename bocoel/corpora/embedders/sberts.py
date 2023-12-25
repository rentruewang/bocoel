import typing
from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from bocoel.corpora.interfaces import Embedder


class SBertEmbedder(Embedder):
    """
    Sentence-BERT embedder. Uses the sentence_transformers library.
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self._sbert = SentenceTransformer(model_name, device=device)

        self.batch_size = batch_size

    def dims(self) -> int:
        d = self._sbert.get_sentence_embedding_dimension()
        assert isinstance(d, int)
        return d

    def _encode(self, text: str | Sequence[str]) -> NDArray:
        if isinstance(text, str):
            text = [text]

        text = list(text)

        result = np.concatenate(
            [
                self._encode_one(text[idx : idx + self.batch_size])
                for idx in range(0, len(text), self.batch_size)
            ]
        )

        assert len(result) == len(text)

        return np.squeeze(result)

    def _encode_one(self, text: List[str]) -> NDArray:
        enc = self._sbert.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return typing.cast(NDArray, enc)
