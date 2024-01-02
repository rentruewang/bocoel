from typing import TypedDict

from ax.modelbridge import Models
from ax.modelbridge.generation_strategy import GenerationStep

_MODEL_MAPPING = {"sobol": Models.SOBOL, "gpmes": Models.GPMES}


class GenStepDict(TypedDict):
    model: str
    num_trials: int


def generation_step(step: GenStepDict | GenerationStep) -> GenerationStep:
    if isinstance(step, GenerationStep):
        return step

    # Copy the dictionary to prevent any potential trouble.
    copied = {**step}
    copied["model"] = _MODEL_MAPPING[copied["model"]]
    return GenerationStep(**copied)
