from enum import Enum


class Task(str, Enum):
    EXPLORE = "explore"
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
