import numpy as np

from bocoel.corpora import Corpus


def check_bounds(corpus: Corpus) -> None:
    bounds = corpus.index.bounds

    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("The bound is not valid")

    lower, upper = bounds.T
    if np.any(lower > upper):
        raise ValueError("lower > upper at some points")
