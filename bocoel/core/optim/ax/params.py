from typing import Any, TypedDict

from typing_extensions import NotRequired

from bocoel.corpora import Index


class AxServiceParameter(TypedDict):
    name: str
    type: str
    bounds: tuple[float, float]
    value_type: NotRequired[str]
    log_scale: NotRequired[bool]


# Currently using Any to silence typing warnings.
def configs(index: Index) -> list[dict[str, Any]]:
    return [name_dict(index, i) for i in range(index.dims)]


def name_dict(index: Index, i: int) -> dict[str, Any]:
    return {
        "name": name(i),
        "type": "range",
        "bounds": index.bounds[i].tolist(),
        "value_type": "float",
    }


def name_list(total: int) -> list[str]:
    return [name(i) for i in range(total)]


def name(number: int) -> str:
    return f"x{number}"
