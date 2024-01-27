from collections.abc import Mapping, Sequence
from typing import Any

import typeguard
from numpy.typing import NDArray

from bocoel.models.adaptors.interfaces import Adaptor
from bocoel.models.lms import LanguageModel


class GlueAdaptor(Adaptor):
    """
    The adaptor for the glue dataset provided by setfit.
    """

    def __init__(
        self,
        idx: str = "idx",
        texts: str = "text",
        label: str = "label",
        choices: Sequence[str] = ("negative", "positive"),
    ) -> None:
        """
        Parameters
        ----------

        `idx: str = "idx"`
        The name of the column containing the indices.

        `texts: str = "text"`
        The name of the column containing the texts.
        Specify multiple columsn with whitespace in between.
        For example, "text1 text2" would be parsed to ["text1", "text2"].
        If there is more than one column, the texts will be concatenated with [SEP].

        `label: str = "label"`
        The name of the column containing the labels.
        """

        self.idx = idx
        self.texts = texts.split()
        self.label = label

        self.choices = choices

    def evaluate(
        self, data: Mapping[str, Sequence[Any]], lm: LanguageModel
    ) -> Sequence[float] | NDArray:
        idx = data[self.idx]
        texts = [data[text] for text in self.texts]
        labels = data[self.label]

        typeguard.check_type("idx", idx, Sequence[int])
        typeguard.check_type("texts", texts, Sequence[Sequence[str]])
        typeguard.check_type("labels", labels, Sequence[int])

        if not all(0 <= i < len(self.choices) for i in labels):
            raise ValueError("labels must be in range [0, choices)")

        sentences = [" [SEP] ".join(txt) for txt in zip(*texts)]
        classified = lm.classify(sentences, choices=self.choices)
        return [float(c == l) for c, l in zip(classified.argmax(-1), labels)]
