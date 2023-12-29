import dataclasses as dcls

from bocoel.core.interfaces import Core, Optimizer
from bocoel.corpora import Corpus
from bocoel.models import Evaluator, LanguageModel


@dcls.dataclass(frozen=True)
class ComposedCore(Core):
    """
    Simply a collection of components.
    """

    corpus: Corpus
    lm: LanguageModel
    evaluator: Evaluator
    optimizer: Optimizer
