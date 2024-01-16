from collections.abc import Callable, Sequence
from typing import Literal, TypedDict

from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.utils import validation
from typing_extensions import NotRequired

from bocoel.corpora import Index, SearchResult

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
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
        model_kwargs: KMeansOptions,
    ) -> None:
        super().__init__(index=index, evaluate_fn=evaluate_fn)

        self._model = KMeans(**model_kwargs)
        self._model.fit(index.embeddings)
        validation.check_is_fitted(self._model)
