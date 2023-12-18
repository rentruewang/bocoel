import typing
from typing import Sequence

from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from bocoel.corpora.interfaces import Embedder


class SBertEmbedder(Embedder):
    def __init__(
        self, model_name: str = "all-mpnet-base-v2", device: str = "cpu"
    ) -> None:
        self._sbert = SentenceTransformer(model_name, device=device)
        dims = self._sbert.get_sentence_embedding_dimension()
        assert isinstance(dims, int)
        self.dims = dims

    def encode(self, text: str | Sequence[str]) -> NDArray:
        if not isinstance(text, str):
            text = list(text)
        return typing.cast(NDArray, self._sbert.encode(text))
