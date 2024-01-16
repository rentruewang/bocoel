from collections.abc import Callable, Sequence
from typing import Literal, TypedDict

from numpy.typing import NDArray
from sklearn.utils import validation
from typing_extensions import NotRequired

from bocoel.corpora import Index, SearchResult

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
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
        model_kwargs: KMedoidsOptions,
    ) -> None:
        # Optional dependency.
        from sklearn_extra.cluster import KMedoids

        super().__init__(index=index, evaluate_fn=evaluate_fn)

        self._model = KMedoids(**model_kwargs)
        self._model.fit(index.embeddings)
        validation.check_is_fitted(self._model)
