from bocoel import (
    BigBenchAdaptor,
    BigBenchMatchType,
    BigBenchQuestionAnswer,
    ExactMatch,
    GenerativeModel,
    NltkBleuScore,
    RougeScore,
    RougeScore2,
    SacreBleuScore,
    Score,
)
from tests import utils


def sacre_bleu_eval(lm: GenerativeModel) -> BigBenchAdaptor:
    return BigBenchQuestionAnswer(
        lm=lm,
        inputs="question",
        targets="answer",
        matching_type=BigBenchMatchType.SACRE_BLEU,
    )


def nltk_bleu_eval(lm: GenerativeModel) -> BigBenchAdaptor:
    return BigBenchQuestionAnswer(
        lm=lm,
        inputs="question",
        targets="answer",
        matching_type=BigBenchMatchType.NLTK_BLEU,
    )


def exact_match_eval(lm: GenerativeModel) -> BigBenchAdaptor:
    return BigBenchQuestionAnswer(
        lm=lm,
        inputs="question",
        targets="answer",
        matching_type=BigBenchMatchType.EXACT,
    )


def rouge_1_eval(lm: GenerativeModel) -> BigBenchAdaptor:
    return BigBenchQuestionAnswer(
        lm=lm,
        inputs="question",
        targets="answer",
        matching_type=BigBenchMatchType.ROUGE_1,
    )


def rouge_2_eval(lm: GenerativeModel) -> BigBenchAdaptor:
    return BigBenchQuestionAnswer(
        lm=lm,
        inputs="question",
        targets="answer",
        matching_type=BigBenchMatchType.ROUGE_2,
    )


def rouge_l_eval(lm: GenerativeModel) -> BigBenchAdaptor:
    return BigBenchQuestionAnswer(
        lm=lm,
        inputs="question",
        targets="answer",
        matching_type=BigBenchMatchType.ROUGE_L,
    )


def rouge_score_1_eval(lm: GenerativeModel) -> BigBenchAdaptor:
    return BigBenchQuestionAnswer(
        lm=lm,
        inputs="question",
        targets="answer",
        matching_type=BigBenchMatchType.ROUGE_SCORE_1,
    )


def rouge_score_2_eval(lm: GenerativeModel) -> BigBenchAdaptor:
    return BigBenchQuestionAnswer(
        lm=lm,
        inputs="question",
        targets="answer",
        matching_type=BigBenchMatchType.ROUGE_SCORE_2,
    )


def rouge_score_l_eval(lm: GenerativeModel) -> BigBenchAdaptor:
    return BigBenchQuestionAnswer(
        lm=lm,
        inputs="question",
        targets="answer",
        matching_type=BigBenchMatchType.ROUGE_SCORE_L,
    )


@utils.cache
def bigbench_adaptor(name: str, lm: GenerativeModel) -> BigBenchAdaptor:
    match name:
        case "sacre-bleu":
            return sacre_bleu_eval(lm=lm)
        case "nltk-bleu":
            return nltk_bleu_eval(lm=lm)
        case "rouge-1":
            return rouge_1_eval(lm=lm)
        case "rouge-2":
            return rouge_2_eval(lm=lm)
        case "rouge-l":
            return rouge_l_eval(lm=lm)
        case "rouge-score-1":
            return rouge_score_1_eval(lm=lm)
        case "rouge-score-2":
            return rouge_score_2_eval(lm=lm)
        case "rouge-score-l":
            return rouge_score_l_eval(lm=lm)
        case "exact-match":
            return exact_match_eval(lm=lm)
        case _:
            raise ValueError(f"Unknown adaptor: {name}")


@utils.cache
def score(name: str) -> Score:
    match name:
        case "sacre-bleu":
            return SacreBleuScore()
        case "nltk-bleu":
            return NltkBleuScore()
        case "rouge-1":
            return RougeScore("rouge-1")
        case "rouge-2":
            return RougeScore("rouge-2")
        case "rouge-l":
            return RougeScore("rouge-l")
        case "rouge-score-1":
            return RougeScore2("rouge1")
        case "rouge-score-2":
            return RougeScore2("rouge2")
        case "rouge-score-l":
            return RougeScore2("rougeL")
        case "exact-match":
            return ExactMatch()
        case _:
            raise ValueError(f"Unknown score: {name}")
