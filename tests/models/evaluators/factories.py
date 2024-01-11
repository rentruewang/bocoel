from bocoel import BigBenchMatchType, BigBenchQuestionAnswer, Evaluator
from tests.models.lms import factories as lm_factories


def bleu() -> Evaluator:
    return BigBenchQuestionAnswer(
        inputs="question", targets="answer", matching_type=BigBenchMatchType.NLTK_BLEU
    )
