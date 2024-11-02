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

    Glue is a collection of datasets for natural language understanding tasks.
    The datasets are designed to be challenging and diverse,
    and they are collected from a variety of sources.
    They are mostly sentence-level classification tasks.

    This adaptor is compatible with all classifier models,
    and it is designed to work with the glue dataset
    (in the format of setfit datasets on huggingface datasets).

    Setfit datasets have the following columns:

    - text: The text to classify.
    - label: The label of the text.
    - label_text: The text of the label.
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
        Initialize the adaptor.

        Parameters:
            lm: The language model to use for classification.
            texts: The column name for the text to classify.
            label: The column name for the label of the text.
            label_text: The column name for the text of the label.
            choices: The valid choices for the label.
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

        typeguard.check_type(texts, Sequence[Sequence[str]])
        typeguard.check_type(labels, Sequence[int])

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
        """
        Get the valid choices for a particular task and split.

        Parameters:
            name: The name of the task.
            split: The split of the task.

        Returns:
            The valid choices for the task and split.
        """

        LOGGER.debug("Getting choices for task", task=name)

        # Perform checks for supported kinds of datasets.
        match name:
            case "sst2" | "mrpc" | "mnli" | "qqp" | "rte" | "qnli":
                pass
            case _:
                raise ValueError(f"Unknown task name {name}")

        # Perform checks for supported kinds of splits.
        match split:
            case "train" | "validation" | "test":
                pass
            case _:
                raise ValueError(f"Unknown split {split}")

        # The actual mux.
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

        raise RuntimeError("Unreachable")
