from bocoel import BigBenchMatchType, BigBenchQuestionAnswer, Evaluator


def sacre_bleu() -> Evaluator:
    return BigBenchQuestionAnswer(
        inputs="question", targets="answer", matching_type=BigBenchMatchType.SACRE_BLEU
    )


def nltk_bleu() -> Evaluator:
    return BigBenchQuestionAnswer(
        inputs="question", targets="answer", matching_type=BigBenchMatchType.NLTK_BLEU
    )
