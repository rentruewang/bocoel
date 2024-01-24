from pathlib import Path

from bocoel.factories import (
    AdaptorName,
    CorpusName,
    EmbedderName,
    IndexName,
    LMName,
    OptimizerName,
    StorageName,
)

from . import data, run


def main(
    *,
    embedder_name: str | EmbedderName = EmbedderName.SBERT,
    embedder_kwargs: str | Path | None = None,
    index_name: str | IndexName = IndexName.WHITENING,
    index_kwargs: str | Path | None = None,
    storage_name: str | StorageName = StorageName.DATASETS,
    storage_kwargs: str | Path | None = None,
    corpus_name: str | CorpusName = CorpusName.COMPOSED,
    adaptor_name: str | AdaptorName = AdaptorName.BIGBENCH_MC,
    adaptor_kwargs: str | Path | None = None,
    lm_name: str | LMName = LMName.LOGITS,
    lm_kwargs: str | Path | None = None,
    optimizer_name: str | OptimizerName = OptimizerName.AX_SERVICE,
    optimizer_kwargs: str | Path | None = None,
    iterations: int = 60,
) -> None:
    # FIXME: Provide actual defaults.
    # Use default configuration if the keyword is not given.
    loaded_embedder_kwargs = data.load(embedder_kwargs) or {}
    loaded_index_kwargs = data.load(index_kwargs) or {}
    loaded_storage_kwargs = data.load(storage_kwargs) or {}
    loaded_adaptor_kwargs = data.load(adaptor_kwargs) or {}
    loaded_lm_kwargs = data.load(lm_kwargs) or {}
    loaded_optimizer_kwargs = data.load(optimizer_kwargs) or {}

    run.with_kwargs(
        embedder_name=embedder_name,
        embedder_kwargs=loaded_embedder_kwargs,
        index_name=index_name,
        index_kwargs=loaded_index_kwargs,
        storage_name=storage_name,
        storage_kwargs=loaded_storage_kwargs,
        corpus_name=corpus_name,
        adaptor_name=adaptor_name,
        adaptor_kwargs=loaded_adaptor_kwargs,
        lm_name=lm_name,
        lm_kwargs=loaded_lm_kwargs,
        optimizer_name=optimizer_name,
        optimizer_kwargs=loaded_optimizer_kwargs,
        iterations=iterations,
    )
