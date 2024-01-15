from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from bocoel.factories import (
    CorpusName,
    EmbedderName,
    EvalName,
    IndexName,
    LMName,
    OptimizerName,
    StorageName,
)

from . import data, run

_USE_DEFAULT_CONFIG: Mapping[str, Any] = MappingProxyType({})


def main(
    *,
    embedder_name: str | EmbedderName = EmbedderName.SBERT,
    embedder_kwargs: str | Path | Mapping[str, Any] = _USE_DEFAULT_CONFIG,
    index_name: str | IndexName = IndexName.WHITENING,
    index_kwargs: str | Path | Mapping[str, Any] = _USE_DEFAULT_CONFIG,
    storage_name: str | StorageName = StorageName.DATASETS,
    storage_kwargs: str | Path | Mapping[str, Any] = _USE_DEFAULT_CONFIG,
    corpus_name: str | CorpusName = CorpusName.COMPOSED,
    evaluator_name: str | EvalName = EvalName.BIGBENCH_MC,
    evaluator_kwargs: str | Path | Mapping[str, Any] = _USE_DEFAULT_CONFIG,
    lm_name: str | LMName = LMName.HUGGINGFACE,
    lm_kwargs: str | Path | Mapping[str, Any] = _USE_DEFAULT_CONFIG,
    optimizer_name: str | OptimizerName = OptimizerName.AX_SERVICE,
    optimizer_kwargs: str | Path | Mapping[str, Any] = _USE_DEFAULT_CONFIG,
    iterations: int = 60,
) -> None:
    embedder_kwargs = data.load(embedder_kwargs)
    index_kwargs = data.load(index_kwargs)
    storage_kwargs = data.load(storage_kwargs)
    evaluator_kwargs = data.load(evaluator_kwargs)
    lm_kwargs = data.load(lm_kwargs)
    optimizer_kwargs = data.load(optimizer_kwargs)

    run.with_kwargs(
        embedder_name=embedder_name,
        embedder_kwargs=embedder_kwargs,
        index_name=index_name,
        index_kwargs=index_kwargs,
        storage_name=storage_name,
        storage_kwargs=storage_kwargs,
        corpus_name=corpus_name,
        evaluator_name=evaluator_name,
        evaluator_kwargs=evaluator_kwargs,
        lm_name=lm_name,
        lm_kwargs=lm_kwargs,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        iterations=iterations,
    )
