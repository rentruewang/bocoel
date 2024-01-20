from typing import Any, TypedDict

from numpy.typing import NDArray
from typing_extensions import NotRequired


class AxServiceParameter(TypedDict):
    name: str
    type: str
    bounds: tuple[float, float]
    value_type: NotRequired[str]
    log_scale: NotRequired[bool]


# Currently using Any to silence typing warnings.
def configs(bounds: NDArray) -> list[dict[str, Any]]:
    return [name_dict(bounds, i) for i in range(len(bounds))]


def name_dict(bounds: NDArray, i: int) -> dict[str, Any]:
    return {
        "name": name(i),
        "type": "range",
        "bounds": bounds[i].tolist(),
        "value_type": "float",
    }


def name_list(total: int) -> list[str]:
    return [name(i) for i in range(total)]


def name(number: int) -> str:
    return f"x{number}"
