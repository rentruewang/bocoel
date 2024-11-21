# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from bocoel import Embedder, EnsembleEmbedder, HuggingfaceEmbedder, SbertEmbedder
from bocoel.common import StrEnum

from . import common


class EmbedderName(StrEnum):
    """
    The names of the embedders.
    """

    SBERT = "SBERT"
    "Corresponds to `SbertEmbedder`."

    HUGGINGFACE = "HUGGINGFACE"
    "Corresponds to `HuggingfaceEmbedder`."

    HUGGINGFACE_ENSEMBLE = "HUGGINGFACE_ENSEMBLE"
    "Corresponds to `EnsembleEmbedder` concatenating `HuggingfaceEmbedder`."


def embedder(
    name: str | EmbedderName,
    /,
    *,
    model_name: str | list[str],
    device: str = "auto",
    batch_size: int,
) -> Embedder:
    """
    Create an embedder.

    Parameters:
        name: The name of the embedder.
        model_name: The model name to use.
        device: The device to use.
        batch_size: The batch size to use.

    Returns:
        The embedder instance.

    Raises:
        ValueError: If the name is unknown.
        TypeError: If the model name is not a string for SBERT or Huggingface,
            or not a list of strings for HuggingfaceEnsemble.
    """

    match EmbedderName.lookup(name):
        case EmbedderName.SBERT:
            if not isinstance(model_name, str):
                raise TypeError(
                    "SbertEmbedder requires a single model name. "
                    f"Got {model_name} instead."
                )

            return common.correct_kwargs(SbertEmbedder)(
                model_name=model_name,
                device=common.auto_device(device),
                batch_size=batch_size,
            )
        case EmbedderName.HUGGINGFACE:
            if not isinstance(model_name, str):
                raise TypeError(
                    "HuggingfaceEmbedder requires a single model name. "
                    f"Got {model_name} instead."
                )
            return common.correct_kwargs(HuggingfaceEmbedder)(
                path=model_name,
                device=common.auto_device(device),
                batch_size=batch_size,
            )
        case EmbedderName.HUGGINGFACE_ENSEMBLE:
            if not isinstance(model_name, list):
                raise TypeError(
                    "HuggingfaceEnsembleEmbedder requires a list of model names. "
                    f"Got {model_name} instead."
                )

            device_list = common.auto_device_list(device, len(model_name))
            return common.correct_kwargs(EnsembleEmbedder)(
                [
                    HuggingfaceEmbedder(path=model, device=dev, batch_size=batch_size)
                    for model, dev in zip(model_name, device_list)
                ]
            )
        case _:
            raise ValueError(f"Unknown embedder name: {name}")
