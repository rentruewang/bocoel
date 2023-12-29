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
    return [param_name_dict(corpus, i) for i in range(corpus.searcher.dims)]


def param_name_dict(corpus: Corpus, i: int) -> dict[str, Any]:
    return {
        "name": param_name(i),
        "type": "range",
        "bounds": corpus.searcher.bounds[i].tolist(),
        "value_type": "float",
    }


def parameter_name_list(total: int) -> list[str]:
    return [param_name(i) for i in range(total)]


def param_name(number: int) -> str:
    return f"x{number}"
