from collections.abc import Mapping, Sequence
from typing import Any, Literal

import structlog
import typeguard
from numpy.typing import NDArray

from bocoel.models.adaptors.interfaces import Adaptor
from bocoel.models.lms import ClassifierModel

LOGGER = structlog.get_logger()


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

        if any(lab not in self.choices for lab in label_texts):
            raise ValueError(
                f"Valid choices={self.choices}. "
                f"Got: {set(lab for lab in label_texts if lab not in self.choices)}"
            )

        if not all(0 <= i < len(self.choices) for i in labels):
            raise ValueError(
                f"labels must be in range [0, {len(self.choices)}),"
                f"because choices={self.choices}"
            )

        sentences = [" [SEP] ".join(txt) for txt in zip(*texts)]
        classified = self.lm.classify(sentences)
        return [float(c == l) for c, l in zip(classified.argmax(-1), labels)]

    @staticmethod
    def task_choices(
        name: Literal["sst2", "mrpc", "mnli", "qqp", "rte", "qnli"],
        split: Literal["train", "validation", "test"],
    ) -> Sequence[str]:
        LOGGER.debug("Getting choices for task", task=name)

        match name, split:
            case "sst2", _:
                return ["negative", "positive"]
            case "mrpc", _:
                return ["not equivalent", "equivalent"]
            # All following cases all use "unlabeled" for "test".
            case _, "test":
                return ["unlabeled"]
            case "mnli", _:
                return ["entailment", "neutral", "contradiction"]
            case "qqp", _:
                return ["not duplicate", "duplicate"]
            case "rte", _:
                return ["entailment", "not entailment"]
            case "qnli", _:
                return ["entailment", "not entailment"]

        raise ValueError(f"Unknown task name {name}")
