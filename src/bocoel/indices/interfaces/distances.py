# Copyright (c) BoCoEL Authors - All Rights Reserved

from bocoel.common import StrEnum

__all__ = ["Distance"]


class Distance(StrEnum):
    """
    Distance metrics.
    """

    L2 = "L2"
    """
    L2 distance. Also known as Euclidean distance.
    """

    INNER_PRODUCT = "IP"
    """
    Inner product distance.
    When normalized, this is equivalent to cosine similarity.
    """
