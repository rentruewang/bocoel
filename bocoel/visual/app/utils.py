# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import copy
from collections.abc import Callable
from typing import ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


def copy_inputs(function: Callable[P, T]) -> Callable[P, T]:
    def copy_input_fn(*args: P.args, **kwargs: P.kwargs) -> T:
        args = copy.deepcopy(args)
        kwargs = copy.deepcopy(kwargs)
        return function(*args, **kwargs)

    return copy_input_fn
