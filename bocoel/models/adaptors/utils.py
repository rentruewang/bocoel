from collections.abc import Mapping, Sequence
from typing import Any


def collate(mappings: Sequence[Mapping[str, Any]]) -> Mapping[str, Sequence[Any]]:
    if len(mappings) == 0:
        return {}

    first = mappings[0]
    keys = first.keys()

    result = {}

    for key in keys:
        extracted = [item[key] for item in mappings]
        result[key] = extracted

    return result
