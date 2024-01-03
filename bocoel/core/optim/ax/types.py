from typing import Any, TypedDict

from typing_extensions import NotRequired

from bocoel.corpora import Searcher


class AxServiceParameter(TypedDict):
    name: str
    type: str
    bounds: tuple[float, float]
    value_type: NotRequired[str]
    log_scale: NotRequired[bool]


# FIXME: Currently using Any to silence typing warnings.
def parameter_configs(searcher: Searcher) -> list[dict[str, Any]]:
    return [param_name_dict(searcher, i) for i in range(searcher.dims)]


def param_name_dict(searcher: Searcher, i: int) -> dict[str, Any]:
    return {
        "name": param_name(i),
        "type": "range",
        "bounds": searcher.bounds[i].tolist(),
        "value_type": "float",
    }


def parameter_name_list(total: int) -> list[str]:
    return [param_name(i) for i in range(total)]


def param_name(number: int) -> str:
    return f"x{number}"
