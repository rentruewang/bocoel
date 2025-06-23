# Copyright (c) BoCoEL Authors - All Rights Reserved

"""
Indices are used for fast nearest neighbor search.
Optionally, they may also perform transformation prior to indexing.

The module provides a few index implementations:

- FaissIndex: Uses the Faiss library for fast nearest neighbor search.
- HnswlibIndex: Uses the hnswlib library for fast nearest neighbor search.
- PolarIndex: Transforms spatial coordinates into polar coordinates for indexing.
- WhiteningIndex: Whitens the data before indexing.
"""

from .backend import *
from .interfaces import *
from .polar import *
from .ppf import *
from .whitening import *
