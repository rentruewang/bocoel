from collections.abc import Mapping, Sequence

from numpy.typing import NDArray

from bocoel.corpora import Corpus
from bocoel.models.scores.interfaces import Score


def collate(mappings: Sequence[Mapping[str, str]]) -> Mapping[str, Sequence[str]]:
    if len(mappings) == 0:
        return {}

    first = mappings[0]
    keys = first.keys()

    result = {}

    for key in keys:
        extracted = [item[key] for item in mappings]
        result[key] = extracted

    return result


def evaluate_on_corpus(
    score: Score, corpus: Corpus, indices: Sequence[int] | NDArray
) -> Sequence[float] | NDArray:
    items = [corpus.storage[idx] for idx in indices]

    collated = collate(items)

    return score.compute(collated)
