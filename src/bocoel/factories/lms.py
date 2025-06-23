# Copyright (c) BoCoEL Authors - All Rights Reserved

from collections.abc import Sequence

from bocoel import (
    ClassifierModel,
    GenerativeModel,
    HuggingfaceGenerativeLM,
    HuggingfaceLogitsLM,
    HuggingfaceSequenceLM,
)

from . import common

__all__ = ["generative", "classifier"]


@common.correct_kwargs
def generative(
    name: str = "HUGGINGFACE_GENERATIVE",
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

    match name:
        case "HUGGINGFACE_GENERATIVE":
            return HuggingfaceGenerativeLM(
                model_path=model_path,
                batch_size=batch_size,
                device=device,
                add_sep_token=add_sep_token,
            )
        case _:
            raise ValueError(f"Unknown LM name {name}")


def classifier(
    name: str,
    /,
    *,
    model_path: str,
    batch_size: int,
    choices: Sequence[str],
    device: str = "auto",
    add_sep_token: bool = False,
) -> ClassifierModel:
    device = common.auto_device(device)

    match name:
        case "HUGGINGFACE_LOGITS":
            return HuggingfaceLogitsLM(
                model_path=model_path,
                batch_size=batch_size,
                device=device,
                choices=choices,
                add_sep_token=add_sep_token,
            )
        case "HUGGINGFACE_SEQUENCE":
            return HuggingfaceSequenceLM(
                model_path=model_path,
                device=device,
                choices=choices,
                add_sep_token=add_sep_token,
            )
        case _:
            raise ValueError(f"Unknown LM name {name}")
