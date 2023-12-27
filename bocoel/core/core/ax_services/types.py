import math
from typing import Any, TypedDict

from typing_extensions import NotRequired

from bocoel.corpora import Corpus


class AxServiceParameter(TypedDict):
    name: str
    type: str
    bounds: tuple[float, float]
    value_type: NotRequired[str]
    log_scale: NotRequired[bool]


# FIXME: Currently using Any to silence typing warnings.
def corpus_parameters(corpus: Corpus) -> list[dict[str, Any]]:
    return [_parameter_dict(corpus, i) for i in range(corpus.index.dims)]


def _parameter_dict(corpus: Corpus, i: int) -> dict[str, Any]:
    dims = corpus.index.dims
    bounds = corpus.index.bounds

    return {
        "name": _parameter_name(i, dims),
        "type": "range",
        "bounds": bounds[i],
        "value_type": "float",
    }


def parameter_name_list(total: int) -> list[str]:
    return [_parameter_name(i, total) for i in range(total)]


def _parameter_name(number: int, total: int) -> str:
    formatted = _format_number_uniform(number, total)
    return f"x{formatted}"


def _format_number_uniform(number: int, total: int) -> str:
    digits = int(math.ceil(math.log10(total)))

    template = "{{:0{}d}}"
    return template.format(digits).format(number)
