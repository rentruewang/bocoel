from collections.abc import Mapping, Sequence
from typing import Any

from numpy.typing import NDArray

from bocoel.models.adaptors.interfaces import Adaptor
from bocoel.models.lms import LanguageModel


class Sst2QuestionAnswer(Adaptor):
    def __init__(
        self,
        idx: str = "idx",
        sentence: str = "sentence",
        label: str = "label",
        choices: int = 2,
    ) -> None:
        self.idx = idx
        self.sentence = sentence
        self.label = label

        self.choices = choices

    def evaluate(
        self, data: Mapping[str, Sequence[Any]], lm: LanguageModel
    ) -> Sequence[float] | NDArray:
        idx = data[self.idx]
        sentences = data[self.sentence]
        labels = data[self.label]

        if not all(isinstance(i, int) for i in idx):
            raise TypeError("idx must be a list of integers")

        if not all(isinstance(i, str) for i in sentences):
            raise TypeError("sentences must be a list of strings")

        if not all(isinstance(i, int) for i in labels):
            raise TypeError("labels must be a list of numbers")

        if not all(0 <= i < self.choices for i in labels):
            raise ValueError("labels must be in range [0, choices)")

        classified = lm.classify(sentences, choices=self.choices)
        return [float(c == l) for c, l in zip(classified.argmax(-1), labels)]
