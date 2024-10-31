import typeguard
from typing import List, Callable, Optional
from .interfaces import Score

class NltkBleuScore(Score):
    def __init__(self, smoothing_function: str = "method7") -> None:
        """
        Initialize the NltkBleuScore class.

        Args:
            smoothing_function (str, optional): The smoothing function to use. Defaults to "method7".
        """
        self.smoothing_function = smoothing_function

    def __call__(self, target: str, references: List[str]) -> float:
        """
        Calculate the BLEU score using NLTK.

        Args:
            target (str): The target sentence.
            references (List[str]): The reference sentences.

        Returns:
            float: The BLEU score.
        """
        typeguard.check_type("references", references, List[str])

        try:
            from nltk.translate import bleu_score
            from nltk.translate.bleu_score import SmoothingFunction
        except ImportError:
            raise ImportError("The NLTK library is required for NltkBleuScore. Please install it using `pip install nltk`.")

        smoothing_functions = {
            "method7": SmoothingFunction().method7,
            "method8": SmoothingFunction().method8,
            # Add more smoothing functions as needed
        }

        if self.smoothing_function not in smoothing_functions:
            raise ValueError(f"Invalid smoothing function: {self.smoothing_function}")

        return bleu_score.sentence_bleu(
            references=[ref.split() for ref in references],
            hypothesis=target.split(),
            smoothing_function=smoothing_functions[self.smoothing_function],
        )

class SacreBleuScore(Score):
    def __init__(self, tokenize: Callable[[str], List[str]] = None) -> None:
        """
        Initialize the SacreBleuScore class.

        Args:
            tokenize (Callable[[str], List[str]], optional): A custom tokenization function. Defaults to None.
        """
        self.tokenize = tokenize

    def __call__(self, target: str, references: List[str]) -> float:
        """
        Calculate the BLEU score using SacreBLEU.

        Args:
            target (str): The target sentence.
            references (List[str]): The reference sentences.

        Returns:
            float: The BLEU score.
        """
        typeguard.check_type("references", references, List[str])

        try:
            from sacrebleu import BLEU
        except ImportError:
            raise ImportError("The sacrebleu library is required for SacreBleuScore. Please install it using `pip install sacrebleu`.")

        if self.tokenize is not None:
            target = self.tokenize(target)
            references = [self.tokenize(ref) for ref in references]

        refs = [[ref] for ref in references]
        return BLEU(
            smooth_method="exp",
            smooth_value=0.0,
            force=False,
            lowercase=False,
            tokenize="intl",
        ).corpus_score(references=refs, hypotheses=[target]).score / 100
