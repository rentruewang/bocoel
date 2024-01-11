from collections.abc import Sequence


class NltkBleuScore:
    def __call__(self, target: str, references: Sequence[str]) -> float:
        # Optional dependency.
        from nltk.translate import bleu_score

        return bleu_score.sentence_bleu(
            references=[ref.split() for ref in references], hypothesis=target.split()
        )
