from typing import Literal, TypedDict

from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.utils import validation
from typing_extensions import NotRequired

from bocoel.core.optim.evals import QueryEvaluator

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
    def __init__(
        self,
        query_eval: QueryEvaluator,
        embeddings: NDArray,
        model_kwargs: KMeansOptions,
    ) -> None:
        super().__init__(query_eval=query_eval)

        self._model = KMeans(**model_kwargs)
        self._model.fit(embeddings)
        validation.check_is_fitted(self._model)
