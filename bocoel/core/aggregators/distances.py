from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numba
from scipy.spatial import distance

from bocoel.core.optim import State
from bocoel.corpora import Corpus
from bocoel.models import Evaluator, LanguageModel

from .interfaces import Agg

ScipyMetric = Literal["euclidean", "cosine"]


class PathLength(Agg):
    def __init__(self, metrics: Sequence[ScipyMetric], /) -> None:
        self._metrics = metrics

    def agg(
        self,
        *,
        corpus: Corpus,
        evaluator: Evaluator,
        lm: LanguageModel,
        states: Sequence[State],
    ) -> Mapping[str, Any]:
        del corpus, evaluator, lm

        return {met: dist(states, met) for met in self._metrics}


@numba.jit(nopython=True)
def dist(states: Sequence[State], metric: ScipyMetric) -> float:
    total = 0
    for source, target in zip(states[:-1], states[1:]):
        total += distance.cdist(source, target, metric=metric)
    return total
