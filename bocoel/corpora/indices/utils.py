from numpy import linalg
from numpy.typing import NDArray


def normalize(embeddings: NDArray, p: int = 2) -> NDArray:
    norm = linalg.norm(embeddings, ord=p, keepdims=True)
    return embeddings / norm
