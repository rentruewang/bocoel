# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from bocoel.common import StrEnum


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
