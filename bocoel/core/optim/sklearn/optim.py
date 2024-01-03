from collections.abc import Callable
from typing import Any

from sklearn.cluster import KMeans
from sklearn.utils import validation
from typing_extensions import Self

from bocoel.core.interfaces import Optimizer, State
from bocoel.core.optim import utils as optim_utils
from bocoel.core.optim.utils import RemainingSteps
from bocoel.corpora import Searcher, SearchResult


# TODO: Add tests.
# TODO: Add other implementations of clustering algorithms.
class SklearnClusterOptimizer(Optimizer):
    """
    The sklearn optimizer that uses clustering algorithms.
    """

    def __init__(
        self,
        searcher: Searcher,
        evaluate_fn: Callable[[SearchResult], float],
        n_clusteres: int,
    ) -> None:
        self._model = KMeans(n_clusters=n_clusteres)
        self._model.fit(searcher.embeddings)
        validation.check_is_fitted(self._model)
        self._remaining_steps = RemainingSteps(n_clusteres)

        self._searcher = searcher
        self._evaluate_fn = evaluate_fn

    @property
    def terminate(self) -> bool:
        return self._remaining_steps.done

    def step(self) -> State:
        self._remaining_steps.step()
        idx = self._remaining_steps.count
        center = self._model.cluster_centers_[idx]

        return optim_utils.evaluate_searcher(
            query=center, searcher=self._searcher, evaluate_fn=self._evaluate_fn
        )

    def render(self, kind: str, **kwargs: Any) -> None:
        raise NotImplementedError

    @classmethod
    def from_searcher(
        cls,
        searcher: Searcher,
        evaluate_fn: Callable[[SearchResult], float],
        **kwargs: Any
    ) -> Self:
        return cls(searcher=searcher, evaluate_fn=evaluate_fn, **kwargs)
