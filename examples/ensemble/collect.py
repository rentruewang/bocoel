import itertools

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
    inc_opts = set(optimizers).difference(non_inc_opts).difference([ground_truth])

    # Last steps, both in time and step index.
    last = last_steps(results)

    ref_emb = reference_embeddings(last, ground_truth)

    non_inc_last_steps = last_for_non_inc(last, non_inc_opts, optimizers)
    inc_results = results[results[columns.OPTIMIZER].isin(inc_opts)]

    experiments = pd.concat([non_inc_last_steps, inc_results])

    plotting_results = metrics(experiments, ref_emb)

    for store, index in itertools.product(storages, indices):
        storage_match = plotting_results[columns.STORAGE] == store
        index_match = plotting_results[columns.INDEX] == index
        data = plotting_results[storage_match & index_match]
        data = data.sort_values(columns.STEP_IDX)

        sns.lineplot(
            data, x=columns.STEP_IDX, y=SPEARMAN_R, hue=columns.OPTIMIZER
        ).set_title(f"{store} - {index}")
        plt.show()


def last_steps(df: DataFrame) -> DataFrame:
    cols = columns.STORAGE, columns.INDEX, columns.MODEL, columns.OPTIMIZER
    return df.sort_values(
        [columns.TIME, columns.STEP_IDX], ascending=False
    ).drop_duplicates(cols)


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

    num_models = len(set(experiments[columns.MODEL]))

    metrics_records = []

    cols = columns.STORAGE, columns.INDEX, columns.OPTIMIZER, columns.STEP_IDX

    for _, (store, idx, opt, step) in experiments[list(cols)].iterrows():
        store_match = experiments[columns.STORAGE] == store
        idx_match = experiments[columns.INDEX] == idx
        opt_match = experiments[columns.OPTIMIZER] == opt
        step_match = experiments[columns.STEP_IDX] == step
        relevant = experiments[store_match & idx_match & opt_match & step_match]

        if len(relevant) != num_models:
            LOGGER.debug(
                "Insufficient models, discarded.",
                rows=len(relevant),
                num_models=num_models,
                storage=store,
                index=idx,
                optimizer=opt,
                step=step,
            )

            continue

        relevant = relevant.sort_values(columns.MODEL)
        spearmanr = stats.spearmanr(
            np.array(relevant[columns.ACC_AVG]), ref_emb[store]
        ).statistic

        metrics_records.append(
            {
                columns.STORAGE: store,
                columns.INDEX: idx,
                columns.OPTIMIZER: opt,
                columns.STEP_IDX: step,
                SPEARMAN_R: spearmanr,
            }
        )

    return DataFrame.from_records(metrics_records)


def last_for_non_inc(
    last_steps: DataFrame, non_inc_opts: list[str], optimizers: list[str]
) -> DataFrame:
    """
    Get the last steps for non incremental optimizers.

    Parameters:
        last_steps: The last steps DataFrame.
        non_inc_opts: The non incremental optimizers.
        optimizers: The optimizers.

    Returns:
        The last steps for non incremental optimizers.
    """

    # Mapping of the parsing from non incremental optimizer
    # to the optimizer and step index.
    non_inc_idx: dict[str, tuple[str, int]] = {}
    for target, optim in itertools.product(non_inc_opts, optimizers):
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

    return non_inc_last_steps.apply(transform, axis=1)


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
    for dataset, index in itertools.product(storages, indices):
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

        references[dataset] = acc_avg

    return references


if __name__ == "__main__":
    sns.set_theme()

    fire.Fire(main)
