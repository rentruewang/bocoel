import itertools

import fire
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from scipy import stats

from bocoel import Manager
from bocoel.core.exams import columns


def main(
    *, path: str = "./results", ground_truth_optimizer: str = "BruteForce()"
) -> None:
    results = Manager.load(path)
    ref_emb = reference_embeddings(results, ground_truth_optimizer)

    storages = sorted(set(results[columns.STORAGE]))
    models = sorted(set(results[columns.MODEL]))
    optimizers = sorted(set(results[columns.OPTIMIZER]))

    if len(storages) != len(ref_emb):
        raise ValueError(
            f"Expected {len(models)} models, got {len(ref_emb)} references."
        )

    # Filter only the latest runs.
    latest_md5 = results.sort_values(columns.TIME, ascending=False).drop_duplicates(
        [columns.MODEL, columns.STORAGE, columns.OPTIMIZER]
    )[columns.MD5]
    results = results[results[columns.MD5].isin(latest_md5)]

    metrics = metric_by_storage_optimizer_steps(results, storages, optimizers, ref_emb)


def metric_by_storage_optimizer_steps(
    results: DataFrame,
    storages: list[str],
    optimizers: list[str],
    ref_emb: dict[str, NDArray],
) -> dict[tuple[str, str, int], float]:
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
