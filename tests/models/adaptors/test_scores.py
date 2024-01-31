import pytest

from tests.corpora.storages import factories as storage_factories

from . import factories


@pytest.mark.parametrize(
    "score_name",
    [
        "sacre-bleu",
        "nltk-bleu",
        "rouge-1",
        "rouge-2",
        "rouge-l",
        "rouge-score-1",
        "rouge-score-2",
        "rouge-score-l",
        "exact-match",
    ],
)
def test_qa_score_on_text(score_name: str) -> None:
    sc = factories.score(score_name)
    df = storage_factories.df()

    question = df["question"]
    answer = df["answer"]

    for q, a in zip(question, answer):
        results = sc(target=q, references=a)
        assert isinstance(results, (int, float))
        assert 0.0 <= results <= 1.0
