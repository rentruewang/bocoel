import itertools
import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

import alive_progress as ap
import fire
import numpy as np
import pandas as pd
import parse
import seaborn as sns
import structlog
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from pandas import DataFrame, Series
from scipy import stats

from bocoel import Manager
from bocoel.core.exams import columns

LOGGER = structlog.get_logger()

SPEARMAN_R = "spearmanr"


def main(
    *,
    path: str = "./results",
    ground_truth: str = "BruteForce()",
    non_incremental: str = "Kmeans({}),KMedoids({})",
) -> None:
    results = Manager.load(path)

    # Get the unique storages and optimizers.
    storages = sorted(set(results[columns.STORAGE]))
    optimizers = sorted(set(results[columns.OPTIMIZER]))
    indices = sorted(set(results[columns.INDEX]))

    # Non incremental and incremental optimizers.
    # Here incremental optimizers means that steps are taken sequentially,
    # whereas non incremental optimizers means that only the last step is used,
    # and that multiple runs are joined together for analysis.
    non_inc_opts = non_incremental.split(",")

    # Last steps, both in time and step index.
    last = last_steps(results)

    ref_emb = reference_embeddings(last, ground_truth)

    non_inc_last_steps, non_inc_keys = last_for_non_inc(last, non_inc_opts, optimizers)
    inc_opts = set(optimizers).difference(non_inc_keys).difference([ground_truth])
    inc_results = results[results[columns.OPTIMIZER].isin(inc_opts)]

    experiments = pd.concat([non_inc_last_steps, inc_results])
    evaluated_metrics = metrics(experiments, ref_emb)

    for store, index in itertools.product(storages, indices):
        storage_match = evaluated_metrics[columns.STORAGE] == store
        index_match = evaluated_metrics[columns.INDEX] == index
        data = evaluated_metrics[storage_match & index_match]

        lineplot(
            data,
            x=columns.STEP_IDX,
            y=SPEARMAN_R,
            hue=columns.OPTIMIZER,
            title=f"{store} - {index}",
        )

    lineplot(
        evaluated_metrics,
        x=columns.STEP_IDX,
        y=SPEARMAN_R,
        hue=columns.OPTIMIZER,
        title="all",
    )


def lineplot(data: DataFrame, x: str, y: str, hue: str, title: str) -> None:
    plt.clf()
    sns.lineplot(data, x=x, y=y, hue=hue).set_title(title)
    fname = re.sub(r"[^\w_. -]", "_", title)
    plt.savefig(f"{fname}.png")


def last_steps(df: DataFrame) -> DataFrame:
    cols = [columns.STORAGE, columns.INDEX, columns.MODEL, columns.OPTIMIZER]
    return (
        df[[*cols, columns.STEP_IDX, columns.ACC_AVG]]
        .groupby(cols, as_index=False)
        .mean()
        .sort_values([columns.STEP_IDX])
        .drop_duplicates(cols)
    )


class ScipyStat:
    def __init__(
        self,
        experiments: DataFrame,
        cols: Sequence[str],
        ref_emb: dict[tuple[str, str], NDArray],
        stat: Literal["spearmanr", "kendalltau"] = "kendalltau",
    ) -> None:
        models = sorted(set(experiments[columns.MODEL]))
        experiments = experiments.sort_values([*cols, columns.MODEL])

        self._groups = experiments.groupby(list(cols))
        self._ref_emb = ref_emb
        self._models = models
        self._stat = stat

        for values in self._ref_emb.values():
            if len(values) != len(models):
                f"Expected {len(models)} models. Got {len(values)}."

    def stat(self, x: NDArray, y: NDArray) -> float:
        match self._stat:
            case "spearmanr":
                return stats.spearmanr(x, y).statistic
            case "kendalltau":
                return stats.kendalltau(x, y).statistic
            case _:
                raise ValueError(f"Unknown statistic {self._stat}.")

    def _compute_one(
        self, grouped: DataFrame, progress: Callable[[], Any]
    ) -> Series | None:
        """
        Calculate the Spearman correlation.
        This function is called `total` times.
        """

        progress()

        assert len(set(grouped[columns.STORAGE])) == 1, "Multiple storages."
        assert len(set(grouped[columns.INDEX])) == 1, "Multiple indices."
        assert len(set(grouped[columns.OPTIMIZER])) == 1, "Multiple optimizers."
        assert len(set(grouped[columns.STEP_IDX])) == 1, "Multiple step indices."

        storage = grouped[columns.STORAGE].iloc[0]
        index = grouped[columns.INDEX].iloc[0]
        optimizer = grouped[columns.OPTIMIZER].iloc[0]
        step_idx = grouped[columns.STEP_IDX].iloc[0]

        grouped = grouped.sort_values(columns.MODEL)
        acc_avg = np.array(grouped[columns.ACC_AVG])
        reference = self._ref_emb[storage, index]

        if len(acc_avg) != len(self._models):
            models = grouped[columns.MODEL]
            LOGGER.debug(
                "Insufficient models, discarded.",
                groups=len(grouped),
                rows=len(acc_avg),
                reference=len(reference),
                storage=storage,
                index=index,
                models=sorted(models),
                all_models=self._models,
            )
            return None

        score = float(stats.kendalltau(acc_avg, reference).statistic)
        return Series(
            {
                columns.STORAGE: storage,
                columns.INDEX: index,
                columns.OPTIMIZER: optimizer,
                columns.STEP_IDX: step_idx,
                SPEARMAN_R: score,
            }
        )

    def compute(self) -> DataFrame:
        """
        Calculate the correlation values from Scipy's registry.

        FIXME:
            DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns

        Returns:
            The correlation values.
        """

        with ap.alive_bar(total=len(self._groups), title="grouping by") as bar:

            def function(grouped: DataFrame):
                return self._compute_one(grouped, bar)

            return self._groups.apply(function)


def metrics(
    experiments: DataFrame, ref_emb: dict[tuple[str, str], NDArray]
) -> DataFrame:
    """
    Calculate the metrics for the experiments.

    Parameters:
        experiments: The experiments DataFrame.
        ref_emb: The reference embeddings.


    Returns:
        The metrics DataFrame.
    """

    cols = columns.STORAGE, columns.INDEX, columns.OPTIMIZER, columns.STEP_IDX

    scipy_stat = ScipyStat(experiments, cols, ref_emb)

    return scipy_stat.compute()


def last_for_non_inc(
    last_steps: DataFrame, non_inc_opts: list[str], optimizers: list[str]
) -> tuple[DataFrame, list[str]]:
    """
    Get the last steps for non incremental optimizers.

    Parameters:
        last_steps: The last steps DataFrame.
        non_inc_opts: The non incremental optimizers.
        optimizers: The optimizers.

    Returns:
        The last steps for non incremental optimizers
        and the non incremental optimizers.
    """

    # Mapping of the parsing from non incremental optimizer
    # to the optimizer and step index.
    non_inc_idx: dict[str, tuple[str, int]] = {}
    for target, optim in ap.alive_it(itertools.product(non_inc_opts, optimizers)):
        if (parsed := parse.parse(target, optim)) is not None:
            idx = int(parsed[0])
            non_inc_idx[optim] = target, idx

    def transform(row: Series) -> Series:
        series = row.copy()

        optimizer = series[columns.OPTIMIZER]
        assert optimizer in non_inc_idx, f"Optimizer {optimizer} not found."

        target, idx = non_inc_idx[optimizer]

        series[columns.OPTIMIZER] = target
        series[columns.STEP_IDX] = idx

        return series

    non_inc_keys = list(non_inc_idx.keys())
    non_inc_last_steps = last_steps[last_steps[columns.OPTIMIZER].isin(non_inc_keys)]

    return non_inc_last_steps.apply(transform, axis=1), non_inc_keys


def reference_embeddings(
    last_steps: DataFrame, truth: str
) -> dict[tuple[str, str], NDArray]:
    """
    Get the reference embeddings for each corpus (storage and index).

    Parameters:
        last_steps: The last steps DataFrame.
        truth: The ground truth optimizer.

    Returns:
        The reference embeddings.
    """

    # Number of unique models and storages.
    indices = sorted(set(last_steps[columns.INDEX]))
    storages = sorted(set(last_steps[columns.STORAGE]))
    num_models = len(set(last_steps[columns.MODEL]))

    # Get the ground truth.
    ground_truth = last_steps[last_steps[columns.OPTIMIZER] == truth]

    if len(ground_truth) != len(storages) * num_models * len(indices):
        raise ValueError(
            f"Expected {len(storages)} * {num_models} * {len(indices)} references, got {len(ground_truth)}."
        )

    # References by storage name.
    references: dict[tuple[str, str], NDArray] = {}
    for dataset, index in ap.alive_it(itertools.product(storages, indices)):
        dataset_match = ground_truth[columns.STORAGE] == dataset
        index_match = ground_truth[columns.INDEX] == index

        matched = ground_truth[dataset_match & index_match]
        matched = matched.sort_values(columns.MODEL)

        acc_avg = np.array(matched[columns.ACC_AVG])

        if len(acc_avg) != num_models:
            raise ValueError(
                f"Expected {num_models} for each storage and index, "
                f"got {len(acc_avg)}."
            )

        references[dataset, index] = acc_avg

    return references


if __name__ == "__main__":
    sns.set_theme()

    fire.Fire(main)
