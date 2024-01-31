from collections.abc import Mapping, Sequence
from typing import Any

import typeguard
from numpy.typing import NDArray
from typing_extensions import Self

from bocoel.models.adaptors.interfaces import Adaptor
from bocoel.models.lms import ClassifierModel


class GlueAdaptor(Adaptor):
    """
    The adaptor for the glue dataset provided by setfit.
    """

    def __init__(
        self,
        lm: ClassifierModel,
        texts: str = "text",
        label: str = "label",
        label_text: str = "label_text",
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

        self.lm = lm

        self.texts = texts.split()
        self.label = label
        self.label_text = label_text
        self.choices = choices

    def evaluate(self, data: Mapping[str, Sequence[Any]]) -> Sequence[float] | NDArray:
        texts = [data[text] for text in self.texts]
        labels = data[self.label]
        label_texts = data[self.label_text]

        typeguard.check_type("texts", texts, Sequence[Sequence[str]])
        typeguard.check_type("labels", labels, Sequence[int])

        choices_set = set(self.choices)
        if not all(lab in choices_set for lab in label_texts):
            raise ValueError("label_texts must not be empty")

        if not all(0 <= i < len(choices_set) for i in labels):
            raise ValueError(
                f"labels must be in range [0, {len(choices_set)}),"
                f"because choices={self.choices}"
            )

        sentences = [" [SEP] ".join(txt) for txt in zip(*texts)]
        classified = self.lm.classify(sentences)
        return [float(c == l) for c, l in zip(classified.argmax(-1), labels)]

    @classmethod
    def task(cls, name: str, model: ClassifierModel) -> Self:
        match name:
            case "sst2":
                return cls(model)
            case "mnli":
                return cls(model, choices=["entailment", "neutral", "contradiction"])
            case "qqp":
                return cls(model, choices=["not duplicate", "duplicate"])
            case "rte":
                return cls(model, choices=["entailment", "not entailment"])
            case "mrpc":
                return cls(model, choices=["not equivalent", "equivalent"])
            case _:
                raise ValueError(f"Unknown task name {name}")
