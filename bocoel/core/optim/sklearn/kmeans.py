# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from typing import Literal, TypedDict

from numpy.typing import NDArray
from sklearn.cluster import KMeans
from typing_extensions import NotRequired

from bocoel.core.optim.interfaces import IndexEvaluator
from bocoel.corpora import Index

from .optim import ScikitLearnOptimizer


class KMeansOptions(TypedDict):
    n_clusters: int
    init: NotRequired[Literal["k-means++", "random"]]
    n_init: NotRequired[int | Literal["auto"]]
    tol: NotRequired[float]
    verbose: NotRequired[int]
    random_state: NotRequired[int]
    algorithm: NotRequired[Literal["llyod", "elkan"]]


class KMeansOptimizer(ScikitLearnOptimizer):
    """
    The KMeans optimizer that uses clustering algorithms.
    """

    def __init__(
        self,
        index_eval: IndexEvaluator,
        index: Index,
        *,
        batch_size: int,
        embeddings: NDArray,
        model_kwargs: KMeansOptions,
    ) -> None:
        """
        Parameters:
            index_eval: The evaluator to use on the storage.
            index: The index to use for the query.
            batch_size: The number of embeddings to evaluate at once.
            embeddings: The embeddings to cluster.
            model_kwargs: The keyword arguments to pass to the KMeans model.
        """

        model = KMeans(**model_kwargs)

        super().__init__(
            index_eval=index_eval,
            index=index,
            embeddings=embeddings,
            model=model,
            batch_size=batch_size,
        )

        self._model_kwargs = model_kwargs

    def __repr__(self) -> str:
        n_clusters = self._model_kwargs["n_clusters"]
        return f"KMeans({n_clusters})"
