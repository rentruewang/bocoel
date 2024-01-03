from collections.abc import Mapping, Sequence

from numpy.typing import NDArray

from bocoel.corpora import Corpus
from bocoel.models.interfaces import Evaluator


def collate(mappings: Sequence[Mapping[str, str]]) -> dict[str, Sequence[str]]:
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
    evaluator: Evaluator, corpus: Corpus, indices: Sequence[int] | NDArray
) -> Sequence[float] | NDArray:
    items = [corpus.storage[idx] for idx in indices]

    collated = collate(items)

    return evaluator.evaluate(collated)
