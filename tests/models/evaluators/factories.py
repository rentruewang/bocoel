from bocoel import BigBenchMatchType, BigBenchQuestionAnswer, Evaluator


def bleu() -> Evaluator:
    return BigBenchQuestionAnswer(
        inputs="question", targets="answer", matching_type=BigBenchMatchType.NLTK_BLEU
    )
