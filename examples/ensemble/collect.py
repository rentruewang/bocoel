import itertools

import fire
import numpy as np
import parse
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from pandas import DataFrame
from scipy import stats

from bocoel import Manager
from bocoel.core.exams import columns


def default_non_inc_opt() -> list[str]:
    return ["Kmeans({})", "KMedoids({})"]


def main(
    *,
    path: str = "./results",
    ground_truth_optimizer: str = "BruteForce()",
    non_incremental_optimizers: list[str] = default_non_inc_opt(),
) -> None:
    sns.set_theme()

    results = Manager.load(path)
    ref_emb = reference_embeddings(results, ground_truth_optimizer)

    storages = sorted(set(results[columns.STORAGE]))
    models = sorted(set(results[columns.MODEL]))
    optimizers = sorted(set(results[columns.OPTIMIZER]))

    if len(storages) != len(ref_emb):
        raise ValueError(
            f"Expected {len(storages)} storages, got {len(ref_emb)} references."
        )

    # Filter only the latest runs.
    latest_md5 = results.sort_values(columns.TIME, ascending=False).drop_duplicates(
        [columns.MODEL, columns.STORAGE, columns.OPTIMIZER]
    )[columns.MD5]
    results = results[results[columns.MD5].isin(latest_md5)]

    metrics = metric_by_storage_optimizer_steps(results, storages, optimizers, ref_emb)

    # Collect the optimizer where only the last step is used together into a single run.
    metrics = collect_non_incremental(metrics, non_incremental_optimizers)

    # Plot results.
    plotting_df = DataFrame(
        [
            {
                "storage": storage,
                "optimizer": optimizer,
                "step": step,
                "spearmanr": spearmanr,
            }
            for (storage, optimizer, step), spearmanr in metrics.items()
            if optimizer != ground_truth_optimizer
        ]
    )

    for store in storages:
        data = plotting_df[plotting_df["storage"] == store]
        sns.lineplot(data, x="step", y="spearmanr", hue="optimizer")
        plt.show()


def collect_non_incremental(
    metrics: dict[tuple[str, str, int], float], non_incremental_optimizers: list[str]
) -> dict[tuple[str, str, int], float]:
    """
    Colllect the optimizers where only the last step makes sense (and is run multiple times).
    """

    # Mapping from the matched items to the tuple of count, template.
    matched: dict[str, tuple[int, str]] = {}

    for _, optimizer, step in metrics.keys():
        for nio in non_incremental_optimizers:
            # Skip if unmatched.
            if parse.parse(nio, optimizer) is None:
                continue

            # Previously matched. Preverse the larger count.
            # Skip if step isn't as large as the previously encountered.
            if optimizer in matched:
                prev_step, _ = matched[optimizer]

                if step < prev_step:
                    continue

            # Matched.
            matched[optimizer] = step, nio

    results: dict[tuple[str, str, int], float] = {}
    for (store, optimizer, step), value in metrics.items():
        # Prevoiusly matched against the template.
        if optimizer in matched:
            matched_step, template = matched[optimizer]

            # Only update if the step matches (largest one).
            if step == matched_step:
                results[store, template, step] = value
        else:
            results[store, optimizer, step] = value

    return results


def metric_by_storage_optimizer_steps(
    results: DataFrame,
    storages: list[str],
    optimizers: list[str],
    ref_emb: dict[str, NDArray],
) -> dict[tuple[str, str, int], float]:
    """
    Calculate the Spearman's R metric for each storage, optimizer and step.

    Parameters:
        results: The results DataFrame.
        storages: The list of storages.
        optimizers: The list of optimizers.
        ref_emb: The reference embeddings.

    Returns:
        A dictionary with the Spearman's R metric for each storage, optimizer and step.
    """

    metrics: dict[tuple[str, str, int], float] = {}

    for st, opt in itertools.product(storages, optimizers):
        storage_match = results[columns.STORAGE] == st
        optimizer_match = results[columns.OPTIMIZER] == opt
        by_st_opt = results[storage_match & optimizer_match]

        min_steps = (
            by_st_opt.sort_values(columns.STEP_IDX, ascending=False)
            .drop_duplicates(columns.MODEL)[columns.STEP_IDX]
            .min()
        )

        by_st_opt = by_st_opt.sort_values(columns.MODEL)
        for step in range(min_steps):
            embeddings = np.array(
                by_st_opt[by_st_opt[columns.STEP_IDX] == step][columns.ACC_AVG]
            )

            storage_ref = ref_emb[st]
            if len(embeddings) != len(storage_ref):
                raise ValueError(
                    f"Expected {len(storage_ref)} embeddings, got {len(embeddings)}."
                )

            statistics = stats.spearmanr(embeddings, storage_ref).statistic
            metrics[st, opt, step] = statistics

            print(
                f"Storage: {st}, Optimizer: {opt}, Step: {step}, Spearman's R: {statistics}"
            )
    return metrics


def reference_embeddings(
    results: DataFrame, ground_truth_optimizer: str
) -> dict[str, NDArray]:
    # Get the ground truth.
    ground_truth = results[results[columns.OPTIMIZER] == ground_truth_optimizer]
    final_step = ground_truth.sort_values(
        columns.STEP_IDX, ascending=False
    ).drop_duplicates([columns.MODEL, columns.STORAGE])

    references = final_step.sort_values(
        [columns.ADAPTOR, columns.STORAGE, columns.MODEL]
    )

    storages = sorted(set(references[columns.STORAGE]))
    models = sorted(set(references[columns.MODEL]))

    if len(references) != len(storages) * len(models):
        raise ValueError(
            f"Expected {len(storages) * len(models)} references, got {len(references)}."
        )

    ref_emb_by_dataset: dict[str, NDArray] = {}
    for dataset in storages:
        current_dataset = references[columns.STORAGE] == dataset
        acc_avg_by_dataset = np.array(references[current_dataset][columns.ACC_AVG])
        ref_emb_by_dataset[dataset] = acc_avg_by_dataset

    return ref_emb_by_dataset


if __name__ == "__main__":
    fire.Fire(main)
