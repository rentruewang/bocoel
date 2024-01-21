from typing import Any, TypedDict

from typing_extensions import NotRequired

from bocoel.corpora import Boundary


class AxServiceParameter(TypedDict):
    name: str
    type: str
    bounds: tuple[float, float]
    value_type: NotRequired[str]
    log_scale: NotRequired[bool]


# Currently using Any to silence typing warnings.
def configs(boundary: Boundary) -> list[dict[str, Any]]:
    return [name_dict(boundary, i) for i in range(boundary.dims)]


def name_dict(boundary: Boundary, i: int) -> dict[str, Any]:
    return {
        "name": name(i),
        "type": "range",
        "bounds": boundary[i].tolist(),
        "value_type": "float",
    }


def name_list(total: int) -> list[str]:
    return [name(i) for i in range(total)]


def name(number: int) -> str:
    return f"x{number}"
