# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from bocoel.core.tasks import Task

from .acquisition import AcquisitionFunc


def check_acquisition_task_combo(acqf: AcquisitionFunc, task: Task) -> None:
    match task, acqf:
        case Task.EXPLORE, AcquisitionFunc.ENTROPY:
            pass
        case Task.EXPLORE, _:
            raise ValueError(
                f"Entropy acquisition function is only supported for {Task.EXPLORE}"
            )
        case (Task.MINIMIZE | Task.MAXIMIZE), AcquisitionFunc.ENTROPY:
            raise ValueError(
                f"Entropy acquisition function is not supported for {task}"
            )
