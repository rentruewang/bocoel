from bocoel import BigBenchEvalutor, BigBenchMatchType, BigBenchQuestionAnswer, Score


def sacre_bleu_eval() -> BigBenchEvalutor:
    return BigBenchQuestionAnswer(
        inputs="question", targets="answer", matching_type=BigBenchMatchType.SACRE_BLEU
    )


def nltk_bleu_eval() -> BigBenchEvalutor:
    return BigBenchQuestionAnswer(
        inputs="question", targets="answer", matching_type=BigBenchMatchType.NLTK_BLEU
    )


def exact_match_eval() -> BigBenchEvalutor:
    return BigBenchQuestionAnswer(
        inputs="question", targets="answer", matching_type=BigBenchMatchType.EXACT
    )


def rouge_1_eval() -> BigBenchEvalutor:
    return BigBenchQuestionAnswer(
        inputs="question", targets="answer", matching_type=BigBenchMatchType.ROUGE_1
    )


def rouge_2_eval() -> BigBenchEvalutor:
    return BigBenchQuestionAnswer(
        inputs="question", targets="answer", matching_type=BigBenchMatchType.ROUGE_2
    )


def rouge_l_eval() -> BigBenchEvalutor:
    return BigBenchQuestionAnswer(
        inputs="question", targets="answer", matching_type=BigBenchMatchType.ROUGE_L
    )


def rouge_score_1_eval() -> BigBenchEvalutor:
    return BigBenchQuestionAnswer(
        inputs="question",
        targets="answer",
        matching_type=BigBenchMatchType.ROUGE_SCORE_1,
    )


def rouge_score_2_eval() -> BigBenchEvalutor:
    return BigBenchQuestionAnswer(
        inputs="question",
        targets="answer",
        matching_type=BigBenchMatchType.ROUGE_SCORE_2,
    )


def rouge_score_l_eval() -> BigBenchEvalutor:
    return BigBenchQuestionAnswer(
        inputs="question",
        targets="answer",
        matching_type=BigBenchMatchType.ROUGE_SCORE_L,
    )


def evaluator(name: str) -> BigBenchEvalutor:
    match name:
        case "sacre_bleu":
            return sacre_bleu_eval()
        case "nltk_bleu":
            return nltk_bleu_eval()
        case "rouge-1":
            return rouge_1_eval()
        case "rouge-2":
            return rouge_2_eval()
        case "rouge-l":
            return rouge_l_eval()
        case "rouge-score-1":
            return rouge_score_1_eval()
        case "rouge-score-2":
            return rouge_score_2_eval()
        case "rouge-score-l":
            return rouge_score_l_eval()
        case "exact_match":
            return exact_match_eval()
        case _:
            raise ValueError(f"Unknown evaluator: {name}")


def score(name: str) -> Score:
    return evaluator(name)._score_fn
