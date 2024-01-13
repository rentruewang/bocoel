from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any, TypeAlias

import numpy as np
from ax.modelbridge import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.service.ax_client import AxClient, ObjectiveProperties
from numpy.typing import NDArray
from torch import device
from typing_extensions import Self

from bocoel.core.optim import utils as optim_utils
from bocoel.core.optim.interfaces import Optimizer, State
from bocoel.corpora import Index, SearchResult

from . import params
from .acquisition import AcquisitionFunc

_KEY = "entropy"
Device: TypeAlias = str | device


class Task(str, Enum):
    EXPLORE = "explore"
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class AxServiceOptimizer(Optimizer):
    """
    The Ax optimizer that uses the service API.
    See https://ax.dev/tutorials/gpei_hartmann_service.html
    """

    def __init__(
        self,
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
        sobol_steps: int,
        device: Device = "cpu",
        workers: int = 1,
        acqf: str | AcquisitionFunc = AcquisitionFunc.MAX_ENTROPY,
        task: Task = Task.EXPLORE,
    ) -> None:
        self._device = device
        self._acqf = AcquisitionFunc(acqf)
        self._task = task

        self._ax_client = AxClient(generation_strategy=self._gen_strat(sobol_steps))
        self._create_experiment(index)

        self._index = index
        self._evaluate_fn = evaluate_fn
        self._workers = workers
        self._terminate = False

    @property
    def terminate(self) -> bool:
        return self._terminate

    def step(self) -> Sequence[State]:
        idx_param, done = self._ax_client.get_next_trials(self._workers)

        if done:
            self._terminate = True

        result_states = []
        for trial_index, parameters in idx_param.items():
            result_states.append(self._eval_trial(trial_index, parameters))

        return result_states

    def _create_experiment(self, index: Index) -> None:
        self._ax_client.create_experiment(
            parameters=params.configs(index),
            objectives={
                _KEY: ObjectiveProperties(minimize=self._task == Task.MINIMIZE)
            },
        )

    def _eval_trial(self, trial_index: int, parameters: dict[str, float]) -> State:
        state = self._eval_params(parameters)

        evaluation = state.evaluation

        if self._task == Task.EXPLORE:
            reported_value = 0.0
        else:
            # Average of all the retrieved neighbors if k != 1.
            # No need to average ovre batch size as currently batch = 1.
            reported_value = np.average(evaluation).item()

        self._ax_client.complete_trial(trial_index, raw_data={_KEY: reported_value})

        return state

    def _eval_params(self, parameters: dict[str, float], k: int = 1) -> State:
        index_dims = self._index.dims
        names = params.name_list(index_dims)
        query = [[parameters[name] for name in names]]

        (result,) = optim_utils.evaluate_index(
            query=query, index=self._index, evaluate_fn=self._evaluate_fn, k=k
        )
        return result

    @classmethod
    def from_index(
        cls,
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
        **kwargs: Any,
    ) -> Self:
        return cls(index=index, evaluate_fn=evaluate_fn, **kwargs)

    @staticmethod
    def _terminate_step(steps: list[GenerationStep]) -> int:
        trials = [step.num_trials for step in steps]
        if all(t >= 0 for t in trials):
            return sum(trials)
        else:
            return -1

    def _gen_strat(self, sobol_steps: int) -> GenerationStrategy:
        return GenerationStrategy(
            [
                GenerationStep(model=Models.SOBOL, num_trials=sobol_steps),
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials=-1,
                    model_kwargs={
                        "torch_device": self._device,
                        "botorch_acqf_class": self._acqf.botorch_acqf_class,
                    },
                ),
            ]
        )
