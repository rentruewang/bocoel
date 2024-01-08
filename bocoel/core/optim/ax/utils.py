from typing import TypedDict

from ax.modelbridge import Models
from ax.modelbridge.generation_strategy import GenerationStep
from typing_extensions import NotRequired

_MODEL_MAPPING = {
    "sobol": Models.SOBOL,
    "gpmes": Models.GPMES,
    "modular": Models.BOTORCH_MODULAR,
}


class ModelsDict(TypedDict):
    torch_device: NotRequired[str]


class GenStepDict(TypedDict):
    model: str
    num_trials: int
    model_kwargs: NotRequired[ModelsDict]


def generation_step(step: GenStepDict | GenerationStep) -> GenerationStep:
    if isinstance(step, GenerationStep):
        return step

    # Copy the dictionary to prevent any potential trouble.
    copied = {**step}
    copied["model"] = _MODEL_MAPPING[copied["model"]]
    return GenerationStep(**copied)
