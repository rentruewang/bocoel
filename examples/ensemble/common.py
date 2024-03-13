import math
from collections.abc import Sequence
from typing import Any, Literal

import structlog
from torch import cuda

from bocoel import (
    AcquisitionFunc,
    Adaptor,
    AxServiceOptimizer,
    BruteForceOptimizer,
    CachedIndexEvaluator,
    ComposedCorpus,
    Corpus,
    CorpusEvaluator,
    Distance,
    Embedder,
    EnsembleEmbedder,
    HnswlibIndex,
    HuggingfaceEmbedder,
    Index,
    InverseCDFIndex,
    KMeansOptimizer,
    KMedoidsOptimizer,
    Optimizer,
    PolarIndex,
    RandomOptimizer,
    Storage,
    Task,
    WhiteningIndex,
)

LOGGER = structlog.get_logger()


class CorpusEvaluatorRegistry:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], CachedIndexEvaluator] = {}

    def __call__(self, corpus: Corpus, adaptor: Adaptor) -> CachedIndexEvaluator:
        key = repr(corpus), repr(adaptor)

        if key not in self._cache:
            corpus_eval = CorpusEvaluator(corpus=corpus, adaptor=adaptor)
            cached_corpus_eval = CachedIndexEvaluator(corpus_eval)
            self._cache[key] = cached_corpus_eval

        return self._cache[key]


def ensemble_embedder(embedders: Sequence[str], batch_size: int) -> EnsembleEmbedder:
    LOGGER.info("Creating embedder", embedders=embedders)
    embs = []

    cuda_available = cuda.is_available()
    device_count = cuda.device_count()
    for i, model in enumerate(embedders):
        # Auto cast devices
        if cuda_available:
            hf_device = f"cuda:{i%device_count}"
        else:
            hf_device = "cpu"
        embs.append(
            HuggingfaceEmbedder(path=model, device=hf_device, batch_size=batch_size)
        )
    return EnsembleEmbedder(embs)


def index_backend_and_kwargs(
    name: str, index_threads: int, batch_size: int, reduced: int
) -> tuple[type[Index], dict[str, Any]]:
    match name:
        case "hnswlib":
            return HnswlibIndex, {"threads": index_threads, "batch_size": batch_size}
        case "inverse_cdf":
            return InverseCDFIndex, {
                "inverse_cdf_backend": HnswlibIndex,
                "threads": index_threads,
                "batch_size": batch_size,
            }
        case "polar":
            return PolarIndex, {
                "polar_backend": HnswlibIndex,
                "threads": index_threads,
                "batch_size": batch_size,
            }
        case "whitening":
            return WhiteningIndex, {
                "whitening_backend": HnswlibIndex,
                "reduced": reduced,
                "threads": index_threads,
                "batch_size": batch_size,
            }
        case _:
            raise ValueError(f"Unknown index backend {name}")


def composed_corpus(
    *,
    batch_size: int,
    device: str,
    index_name: str,
    index_threads: int,
    reduced: int,
    sentence: str,
    storage: Storage,
    embedder: Embedder,
) -> ComposedCorpus:
    LOGGER.info(
        "Creating corpus with storage and embedder",
        storage=storage,
        embedder=embedder,
        device=device,
    )

    index_backend, index_kwargs = index_backend_and_kwargs(
        name=index_name,
        index_threads=index_threads,
        batch_size=batch_size,
        reduced=min(reduced, embedder.dims),
    )
    corpus = ComposedCorpus.index_storage(
        storage=storage,
        embedder=embedder,
        keys=sentence.split(),
        index_backend=index_backend,
        concat=" [SEP] ".join,
        distance=Distance.INNER_PRODUCT,
        **index_kwargs,
    )
    return corpus


def optimizer_and_steps(
    optimizer: Literal["ax", "kmeans", "kmedoids", "random", "brute"],
    optimizer_steps: int,
    corpus: Corpus,
    adaptor: Adaptor,
    sobol_steps: int,
    device: str,
    task: str,
    acqf: str,
    batch_size: int,
    corpus_evals: CorpusEvaluatorRegistry,
) -> tuple[Optimizer, int]:
    corpus_eval = corpus_evals(corpus=corpus, adaptor=adaptor)
    optim: Optimizer

    match optimizer:
        case "ax":
            optim = AxServiceOptimizer(
                index_eval=corpus_eval,
                index=corpus.index,
                sobol_steps=sobol_steps,
                device=device,
                task=Task.lookup(task),
                acqf=AcquisitionFunc.lookup(acqf),
            )
        case "kmeans":
            optim = KMeansOptimizer(
                index_eval=corpus_eval,
                index=corpus.index,
                batch_size=batch_size,
                embeddings=corpus.index.data,
                model_kwargs={"n_clusters": optimizer_steps, "n_init": "auto"},
            )
        case "kmedoids":
            optim = KMedoidsOptimizer(
                index_eval=corpus_eval,
                index=corpus.index,
                batch_size=batch_size,
                embeddings=corpus.index.data,
                model_kwargs={"n_clusters": optimizer_steps},
            )
        case "random":
            optim = RandomOptimizer(
                index_eval=corpus_eval,
                index=corpus.index,
                samples=optimizer_steps,
                batch_size=batch_size,
            )
        case "brute":
            optim = BruteForceOptimizer(
                index_eval=corpus_eval,
                index=corpus.index,
                batch_size=batch_size,
                total=optimizer_steps,
            )
        case _:
            raise ValueError(f"Unknown optimizer {optimizer}")

    match optimizer:
        case "brute":
            LOGGER.info("Brute force optimizer optimizes over the whole corpus")
            LOGGER.info(
                "Setting length to the number of embeddings",
                length=len(corpus.index.data),
            )
            optimizer_steps = math.ceil(len(corpus.index.data) / batch_size)
        case "kmeans" | "kmedoids" | "random":
            LOGGER.info(
                "Setting length to the number of clusters divided by batch",
                steps=optimizer_steps,
            )
            optimizer_steps = math.ceil(optimizer_steps / batch_size)

    return optim, optimizer_steps
