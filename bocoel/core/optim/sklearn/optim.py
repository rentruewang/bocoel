from typing import Any

from sklearn.cluster import KMeans
from sklearn.utils import validation

from bocoel.core.interfaces import Optimizer, State
from bocoel.core.optim import utils as optim_utils
from bocoel.core.optim.utils import RemainingSteps
from bocoel.corpora import Corpus
from bocoel.models import Evaluator, LanguageModel


# TODO: Add tests.
class SklearnClusterOptimizer(Optimizer):
    """
    The sklearn optimizer that uses clustering algorithms.
    """

    def __init__(self, corpus: Corpus, n_clusteres: int) -> None:
        self._model = KMeans(n_clusters=n_clusteres)
        self._model.fit_transform(corpus.searcher.embeddings)
        validation.check_is_fitted(self._model)
        self._remaining_steps = RemainingSteps(n_clusteres)

    @property
    def terminate(self) -> bool:
        return self._remaining_steps.done

    def step(self, corpus: Corpus, lm: LanguageModel, evaluator: Evaluator) -> State:
        self._remaining_steps.step()
        idx = self._remaining_steps.count
        center = self._model.cluster_centers_[idx]

        return optim_utils.evaluate_query(
            query=center, corpus=corpus, lm=lm, evaluator=evaluator
        )

    def render(self, kind: str, **kwargs: Any) -> None:
        raise NotImplementedError
