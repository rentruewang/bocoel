# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import logging
from collections.abc import Mapping
from typing import Any

from ax.modelbridge import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.service.ax_client import AxClient, ObjectiveProperties
from torch import device

from bocoel.core.optim.interfaces import IndexEvaluator, Optimizer
from bocoel.core.tasks import Task
from bocoel.corpora import Boundary, Index

from . import params, utils
from .acquisition import AcquisitionFunc
from .surrogates import SurrogateModel, SurrogateOptions

_KEY = "EVAL"
Device = str | device


def silence_ax() -> None:
    # Disable the very verbose logging from Ax.

    ax_service_loggers = [
        key for key in logging.root.manager.loggerDict if key.startswith("ax.service")
    ]

    for logger in ax_service_loggers:
        logging.getLogger(logger).setLevel(logging.WARNING)


class AxServiceOptimizer(Optimizer):
    """
    The Ax optimizer that uses the service API.
    See https://ax.dev/tutorials/gpei_hartmann_service.html
    """

    def __init__(
        self,
        index_eval: IndexEvaluator,
        index: Index,
        *,
        sobol_steps: int = 0,
        device: Device = "cpu",
        task: Task = Task.EXPLORE,
        acqf: str | AcquisitionFunc = AcquisitionFunc.AUTO,
        surrogate: str | SurrogateModel = SurrogateModel.AUTO,
        surrogate_kwargs: SurrogateOptions | None = None,
    ) -> None:
        """
        Parameters:
            index_eval: The evaluator to use for the query.
            index: The index to for querying.
            sobol_steps: The number of steps to use for the Sobol sequence.
            device: The device to use for the optimization.
            task: The task to use for the optimization.
            acqf: The acquisition function to use for the optimization.
            surrogate: The surrogate model to use for the optimization.
            surrogate_kwargs: The keyword arguments to pass to the surrogate model.
        """

        silence_ax()

        acqf = AcquisitionFunc.lookup(acqf)
        task = Task.lookup(task)

        utils.check_acquisition_task_combo(acqf=acqf, task=task)

        self._device = device
        self._acqf = acqf
        self._surrogate = SurrogateModel.lookup(surrogate).surrogate(surrogate_kwargs)
        self._task = task

        self._ax_client = AxClient(generation_strategy=self._gen_strat(sobol_steps))
        self._create_experiment(index.boundary)

        self._index_eval = index_eval
        self._index = index
        self._terminate = False

    def __repr__(self) -> str:
        return f"AxService({self._task}, {self._acqf})"

    @property
    def task(self) -> Task:
        return self._task

    def step(self) -> Mapping[int, float]:
        """
        Optimize one step with the ax optimizer.

        Note:
            Somehow it seems that with recent versions of ``Ax``,
            it would crash when ``workers > 1`` in ``get_next_trials``.

            Therefore, it's removed.

        Raises:
            StopIteration: When there are no more steps.

        Returns:
            The resulting trial-id to the evaluation result.
        """
        if self._terminate:
            raise StopIteration

        idx_param, done = self._ax_client.get_next_trials(1)

        if done:
            self._terminate = True

        return {
            tidx: self._eval_one_query(tidx, parameters)
            for tidx, parameters in idx_param.items()
        }

    def _create_experiment(self, boundary: Boundary) -> None:
        self._ax_client.create_experiment(
            parameters=params.configs(boundary),
            objectives={
                _KEY: ObjectiveProperties(minimize=self._task == Task.MINIMIZE)
            },
        )

    def _eval_one_query(self, tidx: int, parameters: dict[str, float]) -> float:
        names = params.name_list(len(parameters))
        query = [[parameters[name] for name in names]]

        # Since k=1, the first index is the one we want.
        indices = self._index.search(query=query).indices[..., 0]
        value = self._index_eval(indices)[0]

        # # Exploration with a maximization entropy setting means maximizing y=0.
        # if self._task is Task.EXPLORE:
        #     value = 0

        self._ax_client.complete_trial(tidx, raw_data={_KEY: value})

        return value

    def _gen_strat(self, sobol_steps: int) -> GenerationStrategy:
        modular_kwargs: dict[str, Any] = {"torch_device": self._device}

        if (bac := self._acqf.botorch_acqf_class) is not None:
            modular_kwargs.update({"botorch_acqf_class": bac})

        if self._surrogate is not None:
            modular_kwargs.update({"surrogate": self._surrogate})

        return GenerationStrategy(
            [
                GenerationStep(model=Models.SOBOL, num_trials=sobol_steps),
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials=-1,
                    model_kwargs=modular_kwargs,
                ),
            ]
        )

    @staticmethod
    def _terminate_step(steps: list[GenerationStep]) -> int:
        trials = [step.num_trials for step in steps]
        if all(t >= 0 for t in trials):
            return sum(trials)
        else:
            return -1
