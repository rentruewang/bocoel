import numpy as np
from numpy import linalg
from numpy.typing import NDArray


def normalize(embeddings: NDArray, /, p: int = 2) -> NDArray:
    if embeddings.ndim not in [1, 2]:
        raise ValueError(f"Expected embeddings to be 1D or 2D. Got {embeddings.ndim}D.")

    # Axis = -1 works for both 1D and 2D.
    norm = linalg.norm(embeddings, axis=-1, ord=p, keepdims=True)
    return embeddings / norm


def boundaries(embeddings: NDArray, /) -> NDArray:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings to be 2D, got {embeddings.ndim}D.")

    return np.stack([embeddings.min(axis=0), embeddings.max(axis=0)]).T
