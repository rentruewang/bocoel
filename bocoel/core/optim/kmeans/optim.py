import typing
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypedDict

from numpy.typing import NDArray
from typing_extensions import NotRequired, Self

from bocoel.core.optim import utils as optim_utils
from bocoel.core.optim.interfaces import Optimizer, State
from bocoel.corpora import Index, SearchResult


class KmeansOptions(TypedDict):
    n_clusters: int
    init: NotRequired[Literal["k-means++", "random"]]
    n_init: NotRequired[int | Literal["auto"]]
    tol: NotRequired[float]
    verbose: NotRequired[int]
    random_state: NotRequired[int]
    algorithm: NotRequired[Literal["llyod", "elkan"]]


class KMeansOptimizer(Optimizer):
    """
    The sklearn optimizer that uses clustering algorithms.
    See the following webpage for options
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """

    def __init__(
        self,
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
        **model_kwargs: KmeansOptions,
    ) -> None:
        # Optional dependencies.
        from sklearn.cluster import KMeans
        from sklearn.utils import validation

        # FIXME: Seems like there is some issues with Scikit Learn stub files.
        self._model = KMeans(**typing.cast(Any, model_kwargs))
        self._model.fit(index.embeddings)
        validation.check_is_fitted(self._model)

        self._index = index
        self._evaluate_fn = evaluate_fn

    @property
    def terminate(self) -> bool:
        return True

    def step(self) -> State:
        centers = self._model.cluster_centers_

        return optim_utils.evaluate_index(
            query=centers, index=self._index, evaluate_fn=self._evaluate_fn, k=1
        )

    @classmethod
    def from_index(
        cls,
        index: Index,
        evaluate_fn: Callable[[SearchResult], Sequence[float] | NDArray],
        **kwargs: Any,
    ) -> Self:
        return cls(index=index, evaluate_fn=evaluate_fn, **kwargs)
