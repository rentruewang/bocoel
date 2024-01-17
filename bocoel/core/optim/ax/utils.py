from bocoel.core.optim.interfaces import Task

from .acquisition import AcquisitionFunc


def check_acquisition_task_combo(acqf: AcquisitionFunc, task: Task) -> None:
    if task is Task.EXPLORE:
        if acqf is not AcquisitionFunc.ENTROPY:
            raise ValueError(
                f"Entropy acquisition function is only supported for {Task.EXPLORE}."
            )

    if task in [Task.MAXIMIZE, Task.MINIMIZE]:
        if acqf is AcquisitionFunc.ENTROPY:
            raise ValueError(
                f"Entropy acquisition function is not supported for {task}."
            )

    # FIXME: Remove after fixed.
    if acqf is AcquisitionFunc.MES and task is Task.MINIMIZE:
        raise ValueError(
            f"Max value entropy acquisition for minimization doesn't currently work. "
            "See https://github.com/facebook/Ax/issues/2133"
        )
