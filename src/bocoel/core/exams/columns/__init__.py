# Copyright (c) 2024 RenChu Wang - All Rights Reserved

"""
This module contains the columns names used in the manager dataframes,
which correspond to the different components and exams of the system.
"""

from .components import ADAPTOR, EMBEDDER, INDEX, MD5, MODEL, OPTIMIZER, STORAGE, TIME
from .exams import (
    ACC_AVG,
    ACC_MAX,
    ACC_MIN,
    MST_MAX_EDGE_DATA,
    MST_MAX_EDGE_QUERY,
    ORIGINAL,
    SEGREGATION,
    STEP_IDX,
)
