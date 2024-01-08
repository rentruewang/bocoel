from bocoel import BleuEvaluator, Evaluator
from tests.models.lms import factories as lm_factories


def bleu(device: str) -> Evaluator:
    return BleuEvaluator(
        problem="question", answer="answer", lm=lm_factories.lm(device=device)
    )
