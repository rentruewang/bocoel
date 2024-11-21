# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from collections.abc import Sequence

from bocoel import (
    ClassifierModel,
    GenerativeModel,
    HuggingfaceGenerativeLM,
    HuggingfaceLogitsLM,
    HuggingfaceSequenceLM,
)
from bocoel.common import StrEnum

from . import common


class GeneratorName(StrEnum):
    """
    The generator names.
    """

    HUGGINGFACE_GENERATIVE = "HUGGINGFACE_GENERATIVE"
    "Corresponds to `HuggingfaceGenerativeLM`."


class ClassifierName(StrEnum):
    """
    The classifier names.
    """

    HUGGINGFACE_LOGITS = "HUGGINGFACE_LOGITS"
    "Corresponds to `HuggingfaceLogitsLM`."

    HUGGINGFACE_SEQUENCE = "HUGGINGFACE_SEQUENCE"
    "Corresponds to `HuggingfaceSequenceLM`."


def generative(
    name: str | GeneratorName,
    /,
    *,
    model_path: str,
    batch_size: int,
    device: str = "auto",
    add_sep_token: bool = False,
) -> GenerativeModel:
    """
    Create a generative model.

    Parameters:
        name: The name of the model.
        model_path: The path to the model.
        batch_size: The batch size to use.
        device: The device to use.
        add_sep_token: Whether to add the sep token.

    Returns:
        The generative model instance.

    Raises:
        ValueError: If the name is unknown.
    """

    device = common.auto_device(device)

    match GeneratorName.lookup(name):
        case GeneratorName.HUGGINGFACE_GENERATIVE:
            return common.correct_kwargs(HuggingfaceGenerativeLM)(
                model_path=model_path,
                batch_size=batch_size,
                device=device,
                add_sep_token=add_sep_token,
            )
        case _:
            raise ValueError(f"Unknown LM name {name}")


def classifier(
    name: str | ClassifierName,
    /,
    *,
    model_path: str,
    batch_size: int,
    choices: Sequence[str],
    device: str = "auto",
    add_sep_token: bool = False,
) -> ClassifierModel:
    device = common.auto_device(device)

    match ClassifierName.lookup(name):
        case ClassifierName.HUGGINGFACE_LOGITS:
            return common.correct_kwargs(HuggingfaceLogitsLM)(
                model_path=model_path,
                batch_size=batch_size,
                device=device,
                choices=choices,
                add_sep_token=add_sep_token,
            )
        case ClassifierName.HUGGINGFACE_SEQUENCE:
            return common.correct_kwargs(HuggingfaceSequenceLM)(
                model_path=model_path,
                device=device,
                choices=choices,
                add_sep_token=add_sep_token,
            )
        case _:
            raise ValueError(f"Unknown LM name {name}")
