# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from typing import Literal, TypedDict

from numpy.typing import NDArray
from typing_extensions import NotRequired

from bocoel.core.optim.interfaces import IndexEvaluator
from bocoel.corpora import Index

from .optim import ScikitLearnOptimizer


class KMedoidsOptions(TypedDict):
    n_clusters: int
    metrics: NotRequired[str]
    method: NotRequired[Literal["alternate", "pam"]]
    init: NotRequired[Literal["random", "heuristic", "kmedoids++", "build"]]
    max_iter: NotRequired[int]
    random_state: NotRequired[int]


class KMedoidsOptimizer(ScikitLearnOptimizer):
    """
    The KMedoids optimizer that uses clustering algorithms.
    """

    def __init__(
        self,
        index_eval: IndexEvaluator,
        index: Index,
        *,
        batch_size: int,
        embeddings: NDArray,
        model_kwargs: KMedoidsOptions,
    ) -> None:
        """
        Parameters:
            index_eval: The evaluator to use for the index.
            index: The index to use for the query.
            batch_size: The number of embeddings to evaluate at once.
            embeddings: The embeddings to cluster.
            model_kwargs: The keyword arguments to pass to the KMedoids model.
        """

        # Optional dependency.
        from sklearn_extra.cluster import KMedoids

        model = KMedoids(**model_kwargs)

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
        return f"KMedoids({n_clusters})"
