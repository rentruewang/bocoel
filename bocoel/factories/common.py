# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import functools
import inspect
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from torch import cuda

P = ParamSpec("P")
T = TypeVar("T")


def correct_kwargs(function: Callable[P, T]) -> Callable[P, T]:
    """
    Catches TypeError during function call.
    This happens if function arguments don't match the signature.

    Todo:
        Perhaps make it a decorator for classes directly?
    """

    @functools.wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            result = function(*args, **kwargs)
        except TypeError:
            sig = inspect.signature(function)
            args_str = ", ".join(map(str, args))
            kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())

            raise ValueError(
                "Arguments do not match the function signature. "
                f"Required signature: {sig} "
                f"Got args = ({args_str}, {kwargs_str})"
            )
        return result

    return wrapped


def auto_device(device: str, /) -> str:
    if cuda.is_available():
        return "cuda" if device == "auto" else device
    else:
        return "cpu"


def auto_device_list(device: str, num_models: int, /) -> list[str]:
    device_count = cuda.device_count()

    if device_count:
        if device == "auto":
            return [f"cuda:{i%device_count}" for i in range(num_models)]
        else:
            return [device] * num_models
    else:
        return ["cpu"] * num_models
