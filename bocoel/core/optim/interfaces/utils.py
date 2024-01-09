# This file is defined in the interface module
# because it only depends on interfaces.
# And that the factory functions require access to these functions.
from collections.abc import Callable

from bocoel.corpora import Corpus, SearchResult
from bocoel.models import Evaluator, Score, ScoredEvaluator


def evaluate_corpus_from_score(
    *, corpus: Corpus, score: Score
) -> Callable[[SearchResult], float]:
    """
    Evaluates a corpus on a score metric.

    Since the introduction of the Evaluator abstraction,
    this function uses ScoredEvaluator as a thin wrapper to perform evaluation.
    """

    score_eval = ScoredEvaluator(corpus=corpus, score=score)
    return evaluate_with_evaluator(score_eval)


def evaluate_with_evaluator(evaluator: Evaluator, /) -> Callable[[SearchResult], float]:
    """
    Translates evaluator into a callable evaluating the performance of a SearchResult.
    """

    # FIXME: Only supports results for k=1.
    def evaluate_fn(result: SearchResult) -> float:
        return evaluator.evaluate([result.indices.item()])[0]

    return evaluate_fn
