# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from typing import Any, TypedDict

from typing_extensions import NotRequired

from bocoel.corpora import Boundary


class AxServiceParameter(TypedDict):
    """
    The parameter for the AxServiceOptimizer.
    """

    name: str
    "The name of the parameter."

    type: str
    "The type of the parameter."

    bounds: tuple[float, float]
    "The boundaries of the parameter."

    value_type: NotRequired[str]
    "The value type of the parameter."

    log_scale: NotRequired[bool]
    "Whether the parameter is on a log scale."


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
