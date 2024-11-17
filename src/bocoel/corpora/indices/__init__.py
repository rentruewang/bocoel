# Copyright (c) 2024 RenChu Wang - All Rights Reserved

"""
Indices are used for fast nearest neighbor search.
Optionally, they may also perform transformation prior to indexing.

The module provides a few index implementations:

- FaissIndex: Uses the Faiss library for fast nearest neighbor search.
- HnswlibIndex: Uses the hnswlib library for fast nearest neighbor search.
- PolarIndex: Transforms spatial coordinates into polar coordinates for indexing.
- WhiteningIndex: Whitens the data before indexing.
"""

from .backend import FaissIndex, HnswlibIndex
from .interfaces import (
    Boundary,
    Distance,
    Index,
    InternalResult,
    SearchResult,
    SearchResultBatch,
)
from .polar import PolarIndex
from .ppf import InverseCDFIndex
from .whitening import WhiteningIndex

__all__ = [
    "FaissIndex",
    "HnswlibIndex",
    "Boundary",
    "Distance",
    "Index",
    "InternalResult",
    "SearchResult",
    "SearchResultBatch",
    "PolarIndex",
    "InverseCDFIndex",
    "WhiteningIndex",
]
