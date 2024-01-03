from typing import Any, TypedDict

from typing_extensions import NotRequired

from bocoel.corpora import Index


class AxServiceParameter(TypedDict):
    name: str
    type: str
    bounds: tuple[float, float]
    value_type: NotRequired[str]
    log_scale: NotRequired[bool]


# FIXME: Currently using Any to silence typing warnings.
def parameter_configs(index: Index) -> list[dict[str, Any]]:
    return [param_name_dict(index, i) for i in range(index.dims)]


def param_name_dict(index: Index, i: int) -> dict[str, Any]:
    return {
        "name": param_name(i),
        "type": "range",
        "bounds": index.bounds[i].tolist(),
        "value_type": "float",
    }


def parameter_name_list(total: int) -> list[str]:
    return [param_name(i) for i in range(total)]


def param_name(number: int) -> str:
    return f"x{number}"
