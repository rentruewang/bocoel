import typing
from collections.abc import Sequence

from numpy.typing import NDArray

from bocoel.corpora.embedders.interfaces import Embedder


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
        # Optional dependency.
        from sentence_transformers import SentenceTransformer

        self._sbert = SentenceTransformer(model_name, device=device)

        self._batch_size = batch_size

    @property
    def batch(self) -> int:
        return self._batch_size

    @property
    def dims(self) -> int:
        d = self._sbert.get_sentence_embedding_dimension()
        assert isinstance(d, int)
        return d

    def _encode(self, text: Sequence[str]) -> NDArray:
        text = list(text)

        return typing.cast(
            NDArray,
            self._sbert.encode(
                text,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
            ),
        )
