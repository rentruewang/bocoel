# Copyright (c) RenChu Wang - All Rights Reserved

from bocoel.common import StrEnum


class Task(StrEnum):
    EXPLORE = "EXPLORE"
    MINIMIZE = "MINIMIZE"
    MAXIMIZE = "MAXIMIZE"