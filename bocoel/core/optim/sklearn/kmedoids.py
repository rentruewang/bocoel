from typing import Any, Literal, TypedDict

from numpy.typing import NDArray
from sklearn.utils import validation
from typing_extensions import NotRequired, Self

from bocoel.core.optim.evals import QueryEvaluator
from bocoel.corpora import Boundary

from .optim import ScikitLearnOptimizer


class KMedoidsOptions(TypedDict):
    n_clusters: int
    metrics: NotRequired[str]
    method: NotRequired[Literal["alternate", "pam"]]
    init: NotRequired[Literal["random", "heuristic", "kmedoids++", "build"]]
    max_iter: NotRequired[int]
    random_state: NotRequired[int]


class KMedoidsOptimizer(ScikitLearnOptimizer):
    def __init__(
        self,
        query_eval: QueryEvaluator,
        boundary: Boundary,
        *,
        embeddings: NDArray,
        model_kwargs: KMedoidsOptions,
    ) -> None:
        # Optional dependency.
        from sklearn_extra.cluster import KMedoids

        super().__init__(query_eval=query_eval, boundary=boundary)

        self._model = KMedoids(**model_kwargs)
        self._model.fit(embeddings)
        validation.check_is_fitted(self._model)
