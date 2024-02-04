import logging
import math
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import fire
import structlog
from torch import cuda

import bocoel
from bocoel import (
    AcquisitionFunc,
    Adaptor,
    AxServiceOptimizer,
    BruteForceOptimizer,
    ClassifierModel,
    ComposedCorpus,
    DatasetsStorage,
    Distance,
    EnsembleEmbedder,
    GlueAdaptor,
    HnswlibIndex,
    HuggingfaceEmbedder,
    HuggingfaceLogitsLM,
    HuggingfaceSequenceLM,
    Index,
    KMeansOptimizer,
    KMedoidsOptimizer,
    Manager,
    Optimizer,
    PolarIndex,
    RandomOptimizer,
    Sst2QuestionAnswer,
    Task,
    WhiteningIndex,
)

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

LOGGER = structlog.get_logger()


def main(
    *,
    ds_path: Literal[
        "SST2",
        "SetFit/mnli",
        "SetFit/mrpc",
        "SetFit/qnli",
        "SetFit/rte",
        "SetFit/qqp",
        "SetFit/sst2",
    ] = "SST2",
    ds_split: Literal["train", "validation", "test"] = "train",
    llm_model: str = "textattack/roberta-base-SST-2",
    batch_size: int = 16,
    index_name: Literal["hnswlib", "polar", "whitening"] = "hnswlib",
    sobol_steps: int = 5,
    index_threads: int = 8,
    optimizer_steps: int = 60,
    reduced: int = 32,
    device: str = "cpu",
    acqf: str = "ENTROPY",
    task: str = "EXPLORE",
    classification: Literal["logits", "seq"] = "seq",
    optimizer: Literal["ax", "kmeans", "kmedoids", "random", "brute"] = "ax",
    corpus_cache_path: str | Path = "corpus.pickle",
    embedders: Sequence[str] = tuple(
        [
            # "textattack/bert-base-uncased-SST-2",
            #   "textattack/roberta-base-SST-2",
            #   "textattack/albert-base-v2-SST-2",
            #   "textattack/xlnet-large-cased-SST-2",
            #   "textattack/xlnet-base-cased-SST-2",
            #   "textattack/facebook-bart-large-SST-2",
            #   "textattack/distilbert-base-uncased-SST-2",
            "textattack/distilbert-base-cased-SST-2"
        ]
    ),
    manager_path: str = "results",
) -> None:
    # The corpus part

    sentence, label = sentence_label(ds_path)
    corpus_cache_path = Path(corpus_cache_path)
    corpus: ComposedCorpus
    embedder = ensemble_embedder(batch_size=batch_size, embedders=embedders)

    if corpus_cache_path.exists():
        with open(corpus_cache_path, "rb") as f:
            corpus = pickle.load(f)
    else:
        corpus = composed_corpus(
            ds_path=ds_path,
            ds_split=ds_split,
            batch_size=batch_size,
            device=device,
            index_name=index_name,
            index_threads=index_threads,
            reduced=reduced,
            sentence=sentence,
            embedder=embedder,
        )
        with open(corpus_cache_path, "wb") as f:
            pickle.dump(corpus, f)

    # ------------------------
    # The model part

    task_name: Any = ds_path.lower().replace("setfit/", "")
    if task_name not in ["sst2", "mrpc", "mnli", "qqp", "rte", "qnli"]:
        raise ValueError(f"Unknown task {task_name}")

    lm: ClassifierModel
    LOGGER.info(
        "Creating LM with model", model=llm_model, device=device, task=task_name
    )
    match classification:
        case "seq":
            lm = HuggingfaceSequenceLM(
                model_path=llm_model,
                device=device,
                choices=GlueAdaptor.task_choices(task_name, split=ds_split),
            )
        case "logits":
            lm = HuggingfaceLogitsLM(
                model_path=llm_model,
                batch_size=batch_size,
                device=device,
                choices=GlueAdaptor.task_choices(task_name, split=ds_split),
            )
        case _:
            raise ValueError(f"Unknown classification {classification}")

    # ------------------------
    # Adaptor part

    LOGGER.info("Creating adaptor with arguments", sentence=sentence, label=label)
    adaptor: Adaptor
    if "setfit/" in ds_path.lower():
        adaptor = GlueAdaptor(
            lm,
            texts="text" if "sst2" in task_name else "text1 text2",
            choices=GlueAdaptor.task_choices(task_name, split=ds_split),
        )
    elif ds_path == "SST2":
        adaptor = Sst2QuestionAnswer(lm)
    else:
        raise ValueError(f"Unknown dataset {ds_path}")

    # ------------------------
    # The optimizer part.

    LOGGER.info(
        "Creating optimizer with arguments",
        corpus=corpus,
        lm=lm,
        adaptor=adaptor,
        sobol_steps=sobol_steps,
        device=device,
        acqf=acqf,
    )

    optim: Optimizer
    match optimizer:
        case "ax":
            optim = bocoel.evaluate_corpus(
                AxServiceOptimizer,
                corpus=corpus,
                adaptor=adaptor,
                sobol_steps=sobol_steps,
                device=device,
                task=Task.lookup(task),
                acqf=AcquisitionFunc.lookup(acqf),
            )
        case "kmeans":
            optim = bocoel.evaluate_corpus(
                KMeansOptimizer,
                corpus=corpus,
                adaptor=adaptor,
                batch_size=batch_size,
                embeddings=corpus.index.embeddings,
                model_kwargs={"n_clusters": optimizer_steps, "n_init": "auto"},
            )
        case "kmedoids":
            optim = bocoel.evaluate_corpus(
                KMedoidsOptimizer,
                corpus=corpus,
                adaptor=adaptor,
                batch_size=batch_size,
                embeddings=corpus.index.embeddings,
                model_kwargs={"n_clusters": optimizer_steps},
            )
        case "random":
            optim = bocoel.evaluate_corpus(
                RandomOptimizer,
                corpus=corpus,
                adaptor=adaptor,
                samples=optimizer_steps,
                batch_size=batch_size,
            )
        case "brute":
            optim = bocoel.evaluate_corpus(
                BruteForceOptimizer,
                corpus=corpus,
                adaptor=adaptor,
                embeddings=corpus.index.embeddings,
                batch_size=batch_size,
            )
        case _:
            raise ValueError(f"Unknown optimizer {optimizer}")

    match optimizer:
        case "brute":
            LOGGER.info("Brute force optimizer optimizes over the whole corpus")
            LOGGER.info(
                "Setting length to the number of embeddings",
                length=len(corpus.index.embeddings),
            )
            optimizer_steps = math.ceil(len(corpus.index.embeddings) / batch_size)
        case "kmeans" | "kmedoids" | "random":
            LOGGER.info(
                "Setting length to the number of clusters divided by batch",
                steps=optimizer_steps,
            )
            optimizer_steps = math.ceil(optimizer_steps / batch_size)

    manager = Manager(manager_path)
    scores = manager.run(optimizer=optim, corpus=corpus, steps=optimizer_steps)
    manager.save(
        scores,
        optimizer=optim,
        corpus=corpus,
        model=lm,
        adaptor=adaptor,
        embedder=embedder,
    )


def sentence_label(ds_path: str) -> tuple[str, str]:
    if "setfit" in ds_path.lower():
        sentence = "text" if ds_path == "SetFit/sst2" else "text1 text2"
        label = "label"
    elif ds_path == "SST2":
        sentence = "sentence"
        label = "label"
    else:
        raise ValueError(f"Unknown dataset {ds_path}")
    return sentence, label


def ensemble_embedder(embedders: Sequence[str], batch_size: int) -> EnsembleEmbedder:
    LOGGER.info("Creating embedder")
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
    ds_path: str,
    ds_split: str,
    batch_size: int,
    device: str,
    index_name: str,
    index_threads: int,
    reduced: int,
    sentence: str,
    embedder: EnsembleEmbedder,
) -> ComposedCorpus:
    LOGGER.info("Loading datasets...", dataset=ds_path, split=ds_split)
    storage = DatasetsStorage.load(path=ds_path, split=ds_split)

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


if __name__ == "__main__":
    fire.Fire(main)
