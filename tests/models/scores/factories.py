from bocoel import BleuScore, Score
from tests.models.lms import factories as lm_factories


def bleu(device: str) -> Score:
    return BleuScore(
        problem="question", answers="answer", lm=lm_factories.lm(device=device)
    )
