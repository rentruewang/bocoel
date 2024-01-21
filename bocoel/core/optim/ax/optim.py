from collections.abc import Mapping
from typing import Any

from ax.modelbridge import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.service.ax_client import AxClient, ObjectiveProperties
from numpy.typing import NDArray
from torch import device
from typing_extensions import Self

from bocoel.core.optim.evals import QueryEvaluator
from bocoel.core.optim.interfaces import Optimizer, Task
from bocoel.corpora import Boundary

from . import params, utils
from .acquisition import AcquisitionFunc
from .surrogates import SurrogateModel, SurrogateOptions

_KEY = "entropy"
Device = str | device


class AxServiceOptimizer(Optimizer):
    """
    The Ax optimizer that uses the service API.
    See https://ax.dev/tutorials/gpei_hartmann_service.html
    """

    def __init__(
        self,
        query_eval: QueryEvaluator,
        boundary: Boundary,
        *,
        sobol_steps: int = 0,
        device: Device = "cpu",
        workers: int = 1,
        task: Task = Task.EXPLORE,
        acqf: str | AcquisitionFunc = AcquisitionFunc.AUTO,
        surrogate: str | SurrogateModel = SurrogateModel.AUTO,
        surrogate_kwargs: SurrogateOptions | None = None,
    ) -> None:
        acqf = AcquisitionFunc.lookup(acqf)
        task = Task.lookup(task)

        utils.check_acquisition_task_combo(acqf=acqf, task=task)

        self._device = device
        self._acqf = acqf
        self._surrogate = SurrogateModel.lookup(surrogate).surrogate(surrogate_kwargs)
        self._task = task

        self._ax_client = AxClient(generation_strategy=self._gen_strat(sobol_steps))
        self._create_experiment(boundary)

        self._query_eval = query_eval
        self._workers = workers
        self._terminate = False

    @property
    def task(self) -> Task:
        return self._task

    @property
    def terminate(self) -> bool:
        return self._terminate

    def step(self) -> Mapping[int, float]:
        raise NotImplementedError

        # idx_param, done = self._ax_client.get_next_trials(self._workers)

        # if done:
        #     self._terminate = True

        # result_states = []
        # for trial_index, parameters in idx_param.items():
        #     result_states.append(self._eval_trial(trial_index, parameters))
        # return result_states

    def render(self, **kwargs: Any) -> None:
        raise NotImplementedError

    def _create_experiment(self, boundary: Boundary) -> None:
        self._ax_client.create_experiment(
            parameters=params.configs(boundary),
            objectives={
                _KEY: ObjectiveProperties(minimize=self._task == Task.MINIMIZE)
            },
        )

    # def _eval_trial(self, trial_index: int, parameters: dict[str, float]) -> State:
    #     state = self._eval_params(parameters)

    #     evaluation = state.evaluation

    #     if self._task == Task.EXPLORE:
    #         reported_value = 0.0
    #     else:
    #         # Average of all the retrieved neighbors if k != 1.
    #         # No need to average ovre batch size as currently batch = 1.
    #         reported_value = np.average(evaluation).item()

    #     self._ax_client.complete_trial(trial_index, raw_data={_KEY: reported_value})

    #     return state

    # def _eval_params(self, parameters: dict[str, float], k: int = 1) -> State:
    #     index_dims = self._index.dims
    #     names = params.name_list(index_dims)
    #     query = [[parameters[name] for name in names]]

    #     (result,) = optim_utils.evaluate_index(
    #         query=query, index=self._index, evaluate_fn=self._evaluate_fn, k=k
    #     )
    #     return result

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
