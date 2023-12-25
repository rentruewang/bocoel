from typing import Dict

import numpy as np
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient, ObjectiveProperties

from bocoel.core.interfaces import Core, State
from bocoel.corpora import Corpus
from bocoel.models import Evaluator, LanguageModel

from . import types
from .types import AxServiceParameter

_UNCERTAINTY = "uncertainty"


# FIXME:
# Configuration of generation_strategy should be really easy.
# Now it's hard-coded.
# TODO:
# Use BOTORCH_MODULAR so that it runs on GPU.
# It would also allow configuration of surrogate models.
class AxServiceCore(Core):
    def __init__(
        self,
        corpus: Corpus,
        lm: LanguageModel,
        evaluator: Evaluator,
    ) -> None:
        self._corpus = corpus
        self._lm = lm
        self._evaluator = evaluator
        self._ax_client = AxClient(
            generation_strategy=[
                GenerationStrategy(
                    [
                        GenerationStep(
                            model=Models.SOBOL,
                            num_trials=5,
                        ),
                    ],
                    [
                        GenerationStep(
                            model=Models.GPMES,
                            num_trials=-1,
                        ),
                    ],
                )
            ]
        )
        self._create_experiment()

    @property
    def corpus(self) -> Corpus:
        return self._corpus

    @property
    def lm(self) -> LanguageModel:
        return self._lm

    def optimize(self) -> State:
        # FIXME: Currently only supports 1 item evaluation (in the form of float).
        parameters, trial_index = self._ax_client.get_next_trial()
        state = self._evaluate(parameters)
        self._ax_client.complete_trial(
            trial_index, raw_data={_UNCERTAINTY: float(state.scores)}
        )
        return state

    def _evaluate(self, parameters: Dict[str, AxServiceParameter]) -> State:
        index_dims = self.corpus.index.dims
        names = types.parameter_name_list(index_dims)
        query = np.array([parameters[name] for name in names])
        indices: int = self.corpus.index.search(query).item()
        evaluation = self._evaluator.evaluate(self.lm, self.corpus, indices=indices)
        return State(candidates=query.squeeze(), scores=evaluation)

    def _create_experiment(self) -> None:
        self._ax_client.create_experiment(
            parameters=types.corpus_parameters(self._corpus),
            objectives={_UNCERTAINTY: ObjectiveProperties(minimize=True)},
        )
