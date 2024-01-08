from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.service.ax_client import AxClient, ObjectiveProperties
from typing_extensions import Self

from bocoel.core.optim import utils as optim_utils
from bocoel.core.optim.interfaces import Optimizer, State
from bocoel.core.optim.utils import RemainingSteps
from bocoel.corpora import Index, SearchResult

from . import renderers, types, utils
from .types import AxServiceParameter
from .utils import GenStepDict

_KEY = "entropy"


class AxServiceOptimizer(Optimizer):
    """
    The Ax optimizer that uses the service API.
    See https://ax.dev/tutorials/gpei_hartmann_service.html
    """

    def __init__(
        self,
        index: Index,
        evaluate_fn: Callable[[SearchResult], float],
        steps: Sequence[GenStepDict | GenerationStep],
        minimize: bool = True,
    ) -> None:
        gen_steps = [utils.generation_step(step) for step in steps]
        gen_strat = GenerationStrategy(steps=gen_steps)

        self._ax_client = AxClient(generation_strategy=gen_strat)
        self._create_experiment(index, minimize=minimize)
        self._remaining_steps = RemainingSteps(self._terminate_step(gen_steps))

        self._index = index
        self._evaluate_fn = evaluate_fn

    @property
    def terminate(self) -> bool:
        return self._remaining_steps.done

    def step(self) -> State:
        self._remaining_steps.step()

        # FIXME: Currently only supports 1 item evaluation (in the form of float).
        parameters, trial_index = self._ax_client.get_next_trial()
        state = self._evaluate(parameters)
        self._ax_client.complete_trial(
            trial_index, raw_data={_KEY: float(state.evaluation)}
        )
        return state

    def _create_experiment(self, index: Index, minimize: bool) -> None:
        self._ax_client.create_experiment(
            parameters=types.parameter_configs(index),
            objectives={_KEY: ObjectiveProperties(minimize=minimize)},
        )

    def _evaluate(self, parameters: dict[str, AxServiceParameter]) -> State:
        index_dims = self._index.dims
        names = types.parameter_name_list(index_dims)
        query = np.array([parameters[name] for name in names])

        return optim_utils.evaluate_index(
            query=query, index=self._index, evaluate_fn=self._evaluate_fn
        )

    @classmethod
    def from_index(
        cls,
        index: Index,
        evaluate_fn: Callable[[SearchResult], float],
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

    def render(self, kind: str, **kwargs: Any) -> None:
        """
        See https://ax.dev/tutorials/visualizations.html for details.
        """

        func: Callable

        match kind:
            case "interactive":
                func = renderers.render_interactive
            case "static":
                func = renderers.render_static
            case "tradeoff":
                func = renderers.render_tradeoff
            case "cross_validate" | "cv":
                func = renderers.render_cross_validate
            case "slice":
                func = renderers.render_slice
            case "tile":
                func = renderers.render_tile
            case _:
                raise ValueError("Not supported")

        func(ax_client=self._ax_client, metric_name=_KEY, **kwargs)
