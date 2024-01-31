import logging
from collections.abc import Sequence
from typing import Any, Literal

import datasets
import fire
import numpy as np
import structlog
from torch import cuda
from tqdm import tqdm

import bocoel
from bocoel import (
    AcquisitionFunc,
    Adaptor,
    AxServiceOptimizer,
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
    Optimizer,
    PolarIndex,
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
    ] = "SST2",
    ds_split: Literal["train", "validation", "test"] = "train",
    llm_model: str = "textattack/roberta-base-SST-2",
    batch_size: int = 16,
    index_name: Literal["hnswlib", "polar", "whitening"] = "hnswlib",
    sobol_steps: int = 20,
    index_threads: int = 8,
    optimizer_steps: int = 30,
    remains: int = 32,
    device: str = "cpu",
    acqf: str = "ENTROPY",
    task: str = "EXPLORE",
    classification: Literal["logits", "classifier"] = "classifier",
    optimizer: Literal["ax", "kmeans", "kmedoids"] = "ax",
    choices: Sequence[str] = ("negative", "positive"),
) -> None:
    # The corpus part
    LOGGER.info("Loading datasets...", dataset=ds_path, split=ds_split)
    ds = datasets.load_dataset(ds_path)[ds_split]
    storage = DatasetsStorage(ds)

    embedder = ensemble_embedder(batch_size=batch_size)
    sentence, label = sentence_label(ds_path)

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
        remains=min(remains, embedder.dims),
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

    # ------------------------
    # The model part

    lm: ClassifierModel
    LOGGER.info("Creating LM with model", model=llm_model, device=device)
    match classification:
        case "classifier":
            lm = HuggingfaceSequenceLM(
                model_path=llm_model, device=device, choices=choices
            )
        case "logits":
            lm_cls = HuggingfaceLogitsLM
            lm = HuggingfaceLogitsLM(
                model_path=llm_model,
                batch_size=batch_size,
                device=device,
                choices=choices,
            )
        case _:
            raise ValueError(f"Unknown classification {classification}")

    # ------------------------
    # Adaptor part

    LOGGER.info("Creating adaptor with arguments", sentence=sentence, label=label)
    adaptor: Adaptor
    if "setfit/" in ds_path.lower():
        adaptor = GlueAdaptor.task(ds_path.lower().lstrip("setfit"), lm)
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
                lm=lm.classify,
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
                lm=lm,
                adaptor=adaptor,
                batch_size=batch_size,
                embeddings=corpus.index.embeddings,
                model_kwargs={"n_clusters": optimizer_steps, "n_init": "auto"},
            )
        case "kmedoids":
            optim = bocoel.evaluate_corpus(
                KMedoidsOptimizer,
                corpus=corpus,
                lm=lm,
                adaptor=adaptor,
                batch_size=batch_size,
                embeddings=corpus.index.embeddings,
                model_kwargs={"n_clusters": optimizer_steps},
            )

    scores: list[float] = []
    for i in tqdm(range(optimizer_steps)):
        try:
            state = optim.step()
            LOGGER.info("iteration {i}: {state}", i=i, state=state)
            scores.extend(state.values())
        except StopIteration:
            break

    # Performs aggregation here.
    print("average:", np.average(scores))


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


def ensemble_embedder(batch_size: int):
    LOGGER.info("Creating embedder")
    embedders = []

    cuda_available = cuda.is_available()
    device_count = cuda.device_count()
    for i, model in enumerate(
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
    ):
        # Auto cast devices
        if cuda_available:
            hf_device = f"cuda:{i%device_count}"
        else:
            hf_device = "cpu"
        embedders.append(
            HuggingfaceEmbedder(path=model, device=hf_device, batch_size=batch_size)
        )
    return EnsembleEmbedder(embedders)


def index_backend_and_kwargs(
    name: str, index_threads: int, batch_size: int, remains: int
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
                "remains": remains,
                "threads": index_threads,
                "batch_size": batch_size,
            }
        case _:
            raise ValueError(f"Unknown index backend {name}")


if __name__ == "__main__":
    fire.Fire(main)
