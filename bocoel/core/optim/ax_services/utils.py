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


class RemainingSteps:
    def __init__(self, count: int) -> None:
        self._count = count

    def step(self) -> None:
        self._count -= 1

    @property
    def done(self) -> bool:
        # This would never be true if renaming steps if < 0 at first.
        return self._count == 0
