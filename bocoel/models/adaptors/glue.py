from collections.abc import Mapping, Sequence
from typing import Any

from numpy.typing import NDArray

from bocoel.models.adaptors.interfaces import Adaptor
from bocoel.models.lms import LanguageModel


class GlueAdaptor(Adaptor):
    """
    The adaptor for the glue dataset provided by setfit.
    This adaptor assumes that the dataset has the following columns:
    - `idx`: The index of the entry.
    - `sentence`: The sentence to classify.
    - `label`: The label of the sentence.

    Each entry in the dataset must be a single sentence.
    """

    def __init__(
        self,
        idx: str = "idx",
        text_base: str = "text",
        num_texts: int = 1,
        label: str = "label",
        choices: Sequence[str] = ("negative", "positive"),
    ) -> None:
        self.idx = idx
        self.texts: list[str] = [
            f"{text_base}{i}" if i > 1 else text_base for i in range(1, num_texts + 1)
        ]
        self.label = label

        self.choices = choices

    def evaluate(
        self, data: Mapping[str, Sequence[Any]], lm: LanguageModel
    ) -> Sequence[float] | NDArray:
        idx = data[self.idx]
        texts = [data[text] for text in self.texts]
        labels = data[self.label]

        if not all(isinstance(i, int) for i in idx):
            raise TypeError("idx must be a list of integers")

        for txt in texts:
            if not all(isinstance(i, str) for i in txt):
                raise TypeError("sentences must be a list of strings")

        if len(set(len(txt) for txt in texts)) > 1:
            raise ValueError("All texts must have the same length")

        if not all(isinstance(i, int) for i in labels):
            raise TypeError("labels must be a list of numbers")

        if not all(0 <= i < self.choices for i in labels):
            raise ValueError("labels must be in range [0, choices)")

        sentences = [" [SEP] ".join(texts) for texts in zip(*texts)]
        classified = lm.classify(sentences, choices=self.choices)
        return [float(c == l) for c, l in zip(classified.argmax(-1), labels)]
