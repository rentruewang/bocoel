from collections.abc import Mapping, Sequence
from typing import Any

from bocoel.corpora import Corpus
from bocoel.models import Adaptor, LanguageModel

from .interfaces import Agg


class MergeAgg(Agg):
    def __init__(self, aggs: Sequence[Agg] | Mapping[str, Agg]) -> None:
        self._aggs = aggs

    def agg(
        self,
        *,
        corpus: Corpus,
        adaptor: Adaptor,
        lm: LanguageModel,
    ) -> Mapping[str, Any]:
        items: list[tuple[str, Agg]]
        if isinstance(self._aggs, Sequence):
            items = [("", agg) for agg in self._aggs]
        else:
            items = list(self._aggs.items())

        result: dict[str, Any] = {}
        for prefix, agg in items:
            result.update(
                _aggregate(
                    agg=agg,
                    corpus=corpus,
                    adaptor=adaptor,
                    lm=lm,
                    prefix=prefix,
                )
            )
        return result


def _aggregate(
    agg: Agg, corpus: Corpus, adaptor: Adaptor, lm: LanguageModel, prefix: str = ""
) -> Mapping[str, Any]:
    aggregated = agg.agg(corpus=corpus, adaptor=adaptor, lm=lm)

    if not prefix:
        return aggregated

    return {f"{prefix}{key}": value for key, value in aggregated.items()}
