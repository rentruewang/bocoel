from collections.abc import Mapping, Sequence
from typing import Any, Literal

from scipy.spatial import distance

from bocoel.core.evals import State
from bocoel.corpora import Corpus
from bocoel.models import Adaptor, LanguageModel

from .interfaces import Agg

ScipyMetric = Literal["euclidean", "cosine"]


class PathLength(Agg):
    def __init__(self, metrics: Sequence[ScipyMetric], /) -> None:
        self._metrics = metrics

    def agg(
        self,
        *,
        corpus: Corpus,
        adaptor: Adaptor,
        lm: LanguageModel,
        states: Sequence[State],
    ) -> Mapping[str, Any]:
        del corpus, adaptor, lm

        return {met: dist(states, met) for met in self._metrics}


def dist(states: Sequence[State], metric: ScipyMetric) -> float:
    total = 0
    for source, target in zip(states[:-1], states[1:]):
        total += distance.cdist(source, target, metric=metric)
    return total
