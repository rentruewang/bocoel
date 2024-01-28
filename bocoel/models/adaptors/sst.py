from collections.abc import Mapping, Sequence
from typing import Any

import typeguard
from numpy.typing import NDArray

from bocoel.models.adaptors.interfaces import Adaptor
from bocoel.models.lms import LanguageModel


class Sst2QuestionAnswer(Adaptor):
    """
    The adaptor for the SST-2 dataset.
    This adaptor assumes that the dataset has the following columns:
    - `idx`: The index of the entry.
    - `sentence`: The sentence to classify.
    - `label`: The label of the sentence.

    Each entry in the dataset must be a single sentence.
    """

    def __init__(
        self,
        sentence: str = "sentence",
        label: str = "label",
        choices: Sequence[str] = ("negative", "positive"),
    ) -> None:
        self.sentence = sentence
        self.label = label

        self.choices = choices

    def evaluate(
        self, data: Mapping[str, Sequence[Any]], lm: LanguageModel
    ) -> Sequence[float] | NDArray:
        sentences = data[self.sentence]
        labels = data[self.label]

        typeguard.check_type("sentences", sentences, Sequence[str])
        typeguard.check_type("labels", labels, Sequence[int])

        if not all(0 <= i < len(self.choices) for i in labels):
            raise ValueError("labels must be in range [0, choices)")

        classified = lm.classify(sentences, choices=self.choices)
        return [float(c == l) for c, l in zip(classified.argmax(-1), labels)]
