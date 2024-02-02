from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing_extensions import Self

from bocoel.core.optim import Optimizer
from bocoel.corpora import StatefulIndex

from . import statistics
from .statistics import MetricType

TRIAL_ID = "trial_id"
STEP_IDX = "step_idx"

COLUMNS = (
    TRIAL_ID,
    STEP_IDX,
    MetricType.VALUE,
    MetricType.ACC_MIN,
    MetricType.ACC_MAX,
    MetricType.ACC_AVG,
    MetricType.MAX_MST_QUERY,
    MetricType.MAX_MST_DATA,
)


class Manager:
    def __init__(self, df: DataFrame | None = None) -> None:
        if df is None:
            df = DataFrame(columns=COLUMNS)

        self._df = df

    def run_record(
        self, keys: Sequence[str], index: StatefulIndex, optimizer: Optimizer
    ) -> None:
        """
        Run the optimizer and record the states.

        Parameters
        ----------

        `keys : Sequence[str]`
        The keys to use for the results. The keys are to be hashed.

        `index : StatefulIndex`
        The index to retrieve results from.

        `optimizer : Optimizer`
        The optimizer to run.

        `steps : int`
        The number of steps to run the optimizer for.
        """

        results = optimizer.run()
        self.store(keys=keys, index=index, states=results)

    def store(
        self, keys: Sequence[str], index: StatefulIndex, states: OrderedDict[int, float]
    ) -> None:
        """
        Store the states in the manager.

        Parameters
        ----------

        `keys : Sequence[str]`
        The keys to use for the results. The keys are to be hashed.

        `index : StatefulIndex`
        The index to retrieve results from.

        `states : OrderedDict[int, float]`
        The output of the optimizer.
        """

        keys = tuple(keys)
        stats_df = statistics.stats(index=index, states=states)

        # Compute aggregation index to prevent collision between runs.
        stats_df[TRIAL_ID] = [keys] * len(stats_df)
        stats_df[STEP_IDX] = np.arange(len(stats_df))

        self._df = pd.concat([self._df, stats_df])

    def pretty_print(self) -> None:
        print(self._df.to_string())

    def dump(self, path: str | Path) -> None:
        if not isinstance(path, Path):
            path = Path(path)

        match path.suffix:
            case ".csv":
                self._df.to_csv(path, index=False)
            case ".json":
                self._df.to_json(path, index=False)
            case ".xml":
                self._df.to_xml(path, index=False)
            case ".pkl":
                self._df.to_pickle(path, index=False)
            case _:
                raise ValueError(f"Unsupported file format: {path.suffix}")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        if not isinstance(path, Path):
            path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"{path} is not a file.")

        match path.suffix:
            case ".csv":
                df = pd.read_csv(path)
            case ".json":
                df = pd.read_json(path)
            case ".xml":
                df = pd.read_xml(path)
            case ".pkl":
                df = pd.read_pickle(path)
            case _:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        missing_fields = set(COLUMNS).difference(set(df.columns))
        if len(missing_fields) > 0:
            raise ValueError(
                f"Expected dataframe to contain {COLUMNS}, got {df.columns}. "
                f"Missing {missing_fields}."
            )

        return cls(df)
