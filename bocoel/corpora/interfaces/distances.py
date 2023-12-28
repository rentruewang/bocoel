from enum import Enum


class Distance(str, Enum):
    L2 = "l2"
    INNER_PRODUCT = "ip"
