from bocoel.core.tasks import Task

from .acquisition import AcquisitionFunc


def check_acquisition_task_combo(acqf: AcquisitionFunc, task: Task) -> None:
    if task is Task.EXPLORE:
        if acqf is not AcquisitionFunc.ENTROPY:
            raise ValueError(
                f"Entropy acquisition function is only supported for {Task.EXPLORE}"
            )

    if task in [Task.MAXIMIZE, Task.MINIMIZE]:
        if acqf is AcquisitionFunc.ENTROPY:
            raise ValueError(
                f"Entropy acquisition function is not supported for {task}"
            )
