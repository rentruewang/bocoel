import logging
from collections import OrderedDict
from typing import Literal

import fire
import numpy as np
import structlog
from numpy import random
from numpy.typing import ArrayLike

from bocoel import (
    AcquisitionFunc,
    AxServiceOptimizer,
    Boundary,
    Optimizer,
    RandomOptimizer,
    Task,
)

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

LOGGER = structlog.get_logger()


def minmax(
    task: Literal["MININUM", "MAXIMUM"],
    iters: int,
    acqf: str,
    mean: float = 0,
    std: float = 1,
    p: int = 2,
    dims: int = 32,
    sobol_steps: int = 5,
    device: str = "cpu",
    workers: int = 1,
    seed: int = 42,
) -> None:
    main(
        task=task,
        iters=iters,
        acqf=acqf,
        mean=mean,
        std=std,
        p=p,
        dims=dims,
        sobol_steps=sobol_steps,
        device=device,
        workers=workers,
        seed=seed,
    )


def explore(
    iters: int,
    acqf: str,
    mean: float = 0,
    std: float = 1,
    p: int = 2,
    dims: int = 32,
    sobol_steps: int = 5,
    device: str = "cpu",
    workers: int = 1,
    seed: int = 42,
) -> None:
    main(
        task="EXPLORE",
        iters=iters,
        acqf=acqf,
        mean=mean,
        std=std,
        p=p,
        dims=dims,
        sobol_steps=sobol_steps,
        device=device,
        workers=workers,
        seed=seed,
    )


def main(
    task: Literal["MININUM", "MAXIMUM", "EXPLORE"],
    iters: int,
    acqf: str,
    mean: float = 0,
    std: float = 1,
    p: int = 2,
    dims: int = 32,
    sobol_steps: int = 5,
    device: str = "cpu",
    workers: int = 1,
    seed: int = 42,
) -> None:
    random.seed(seed)
    center = random.randn(dims) * std

    def query_eval(query: ArrayLike) -> OrderedDict[int, float]:
        query = np.array(query)
        assert query.ndim == 2, query.shape

        result = (query - center[None, :]) ** p
        average = -np.sum(result, axis=1)

        assert average.shape == (len(query),)
        return OrderedDict(enumerate(average))

    optimizer: Optimizer
    if acqf.upper() == "RANDOM":
        optimizer = RandomOptimizer(
            query_eval=query_eval,
            boundary=Boundary.fixed(lower=mean - std, upper=mean + std, dims=dims),
            samples=iters,
            batch_size=64,
        )

    else:
        optimizer = AxServiceOptimizer(
            query_eval=query_eval,
            boundary=Boundary.fixed(lower=mean - std, upper=mean + std, dims=dims),
            sobol_steps=sobol_steps,
            device=device,
            workers=workers,
            task=Task.lookup(task),
            acqf=AcquisitionFunc.lookup(acqf),
        )

    min_out = float("inf")
    max_out = float("-inf")
    total = 0.0
    count = 0

    for i in range(iters):
        try:
            output = optimizer.step()

            count += len(output)
            total += sum(output.values())
            min_out = min(min_out, min(output.values()))
            max_out = max(max_out, max(output.values()))

            LOGGER.info(
                "Iteration", i=i, min=min_out, max=max_out, average=total / count
            )

        except StopIteration:
            LOGGER.info("Terminated")
            break


if __name__ == "__main__":
    fire.Fire({"minmax": minmax, "explore": explore})
