# Copyright (c) BoCoEL Authors - All Rights Reserved

from bocoel.common import StrEnum

__all__ = ["Task"]


class Task(StrEnum):
    EXPLORE = "EXPLORE"
    MINIMIZE = "MINIMIZE"
    MAXIMIZE = "MAXIMIZE"
