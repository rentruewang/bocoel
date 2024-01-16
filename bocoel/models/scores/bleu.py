from collections.abc import Sequence

from .interfaces import Score


class NltkBleuScore(Score):
    def __call__(self, target: str, references: Sequence[str]) -> float:
        # Optional dependency.
        from nltk.translate import bleu_score
        from nltk.translate.bleu_score import SmoothingFunction

        return bleu_score.sentence_bleu(
            references=[ref.split() for ref in references],
            hypothesis=target.split(),
            smoothing_function=SmoothingFunction().method7,
        )


class SacreBleuScore(Score):
    def __init__(self) -> None:
        # Optional dependency.
        from sacrebleu import BLEU

        self._bleu = BLEU(
            smooth_method="exp",
            smooth_value=0.0,
            force=False,
            lowercase=False,
            tokenize="intl",
        )

    def __call__(self, target: str, references: Sequence[str]) -> float:
        return (
            self._bleu.corpus_score(
                references=[[ref] for ref in references], hypotheses=[target]
            ).score
            / 100
        )
