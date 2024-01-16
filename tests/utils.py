import functools
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from torch import cuda

P = ParamSpec("P")
T = TypeVar("T")


def cache(func: Callable[P, T]) -> Callable[P, T]:
    # Ignore because of current functools bug.
    # https://stackoverflow.com/questions/73517571/typevar-inference-broken-by-lru-cache-decorator
    return functools.cache(func)  # type:ignore


@cache
def faiss():
    # Optional dependency. Faiss also spits out deprecation warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        import faiss

    return faiss


def torch_devices() -> list[str]:
    """
    Avaialble devices that the embedders are supposed to run on.
    If CUDA is available, both CPU and CUDA are tested.
    """

    device_list = ["cpu"]

    if cuda.is_available():
        device_list.append("cuda")

    return device_list


def faiss_devices() -> list[str]:
    """
    Avaialble devices that the embedders are supposed to run on.
    If CUDA is available, both CPU and CUDA are tested.
    """

    device_list = ["cpu"]

    if faiss().get_num_gpus() > 0:
        device_list.append("cuda")

    return device_list
