# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import typeguard

from .interfaces import Score


class NltkBleuScore(Score):
    def __call__(self, target: str, references: list[str]) -> float:
        typeguard.check_type(references, list[str])

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

    def __call__(self, target: str, references: list[str]) -> float:
        typeguard.check_type(references, list[str])

        refs = [[ref] for ref in references]
        return self._bleu.corpus_score(references=refs, hypotheses=[target]).score / 100
