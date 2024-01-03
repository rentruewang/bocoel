import numpy as np
from numpy import linalg
from numpy.typing import NDArray


def validate_embeddings(embeddings: NDArray, /, dims: int | list[int] = 2) -> None:
    if isinstance(dims, int):
        dims = [dims]

    if embeddings.ndim not in dims:
        d_str = " or ".join(f"{i}D" for i in dims)
        raise ValueError(f"Expected embeddings to be {d_str}, got {embeddings.ndim}D")


def normalize(embeddings: NDArray, /, p: int = 2) -> NDArray:
    validate_embeddings(embeddings, [1, 2])

    # Axis = -1 works for both 1D and 2D.
    norm = linalg.norm(embeddings, axis=-1, ord=p, keepdims=True)
    return embeddings / norm


def boundaries(embeddings: NDArray, /) -> NDArray:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings to be 2D, got {embeddings.ndim}D")

    return np.stack([embeddings.min(axis=0), embeddings.max(axis=0)]).T
