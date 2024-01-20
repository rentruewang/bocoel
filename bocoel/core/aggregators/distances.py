from collections.abc import Mapping, Sequence
from typing import Any, Literal

from scipy.spatial import distance

from bocoel.corpora import Corpus, SearchResult
from bocoel.models import Adaptor, LanguageModel

from .interfaces import Agg

ScipyMetric = Literal["euclidean", "cosine"]


class PathLength(Agg):
    def __init__(self, metrics: Sequence[ScipyMetric], /) -> None:
        self._metrics = metrics

    def agg(
        self, *, corpus: Corpus, adaptor: Adaptor, lm: LanguageModel
    ) -> Mapping[str, Any]:
        search_results = corpus.index.history
        return {met: dist(search_results, met) for met in self._metrics}


def dist(states: Sequence[SearchResult], metric: ScipyMetric) -> float:
    total = 0
    for source, target in zip(states[:-1], states[1:]):
        total += distance.cdist(source.query, target.query, metric=metric)
    return total
