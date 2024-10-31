from collections import OrderedDict
from collections.abc import Mapping
from typing import Dict, Any
from pandas import DataFrame
from concurrent.futures import ThreadPoolExecutor
from logging import Logger, getLogger

from bocoel.core.exams.interfaces import Exam
from bocoel.corpora import Index
from .columns import exams
from .stats import AccType, Accumulation

class Examinator:
    """
    The Examinator is responsible for launching exams.
    Examinators take in an index and results of an optimizer run,
    and return a DataFrame of scores for the accumulated history performance of the optimizer.
    """

    def __init__(self, exams: Mapping[str, Exam]) -> None:
        """
        Initialize the Examinator with a mapping of exam names to Exam instances.

        Parameters:
            exams: A mapping of exam names to Exam instances.
        """
        self.exams = exams
        self.logger = getLogger(__name__)

    def examine(self, index: Index, results: OrderedDict[int, float]) -> DataFrame:
        """
        Perform the exams on the results.

        This method looks up results in the index and runs the exams on the results.
        It uses multi-threading to run the exams in parallel.

        Parameters:
            index: The index of the results.
            results: The results.

        Returns:
            The scores of the exams as a DataFrame.

        Raises:
            ValueError: If the input parameters are invalid.
        """
        if not isinstance(index, Index):
            raise ValueError("Invalid index type")
        if not isinstance(results, OrderedDict):
            raise ValueError("Invalid results type")

        # Use ThreadPoolExecutor to run exams in parallel
        with ThreadPoolExecutor() as executor:
            futures = {k: executor.submit(v.run, index, results) for k, v in self.exams.items()}
            scores = {k: future.result() for k, future in futures.items()}

        original = {
            exams.STEP_IDX: list(range(len(results))),
            exams.ORIGINAL: list(results.values()),
        }
        self.logger.info("Exams completed. Creating DataFrame.")
        return DataFrame.from_dict({**original, **scores})

    @classmethod
    def presets(cls) -> "Examinator":
        """
        Returns the default examinator.

        Returns:
            The default Examinator instance.
        """
        return cls(
            {
                exams.ACC_MIN: Accumulation(AccType.MIN),
                exams.ACC_MAX: Accumulation(AccType.MAX),
                exams.ACC_AVG: Accumulation(AccType.AVG),
            }
        )

# Example usage
if __name__ == "__main__":
    from logging import basicConfig, INFO

    basicConfig(level=INFO)
    logger = getLogger(__name__)

    # Assuming Index and Exam instances are created elsewhere
    index = Index()  # Replace with actual Index instance
    results = OrderedDict([(i, i * 2) for i in range(10)])  # Replace with actual results

    examinator = Examinator.presets()
    scores_df = examinator.examine(index, results)
    logger.info("Scores DataFrame:\n%s", scores_df)
