import functools
import inspect
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def correct_kwargs(function: Callable[P, T]) -> Callable[P, T]:
    """
    Catches TypeError during function call.
    This happens if function arguments don't match the signature.

    TODO: Perhaps make it a decorator for classes directly?
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
