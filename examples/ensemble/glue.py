import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import structlog

from bocoel import (
    Adaptor,
    ClassifierModel,
    ComposedCorpus,
    DatasetsStorage,
    Embedder,
    GlueAdaptor,
    HuggingfaceLogitsLM,
    HuggingfaceSequenceLM,
    Manager,
    SbertEmbedder,
    Sst2QuestionAnswer,
)

from . import common
from .common import CorpusEvaluatorRegistry

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
    ds_split: Literal["train", "validation", "test"] = "validation",
    llm_model: str = "textattack/roberta-base-SST-2",
    batch_size: int = 16,
    index_name: Literal["hnswlib", "polar", "whitening", "inverse_cdf"] = "hnswlib",
    sobol_steps: int = 5,
    index_threads: int = 8,
    optimizer: Literal["ax", "kmeans", "kmedoids", "random", "brute"] = "ax",
    optimizer_steps: int = 60,
    reduced: int = 32,
    device: str = "cpu",
    acqf: str = "ENTROPY",
    task: str = "EXPLORE",
    classification: Literal["logits", "seq"] = "seq",
    corpus_cache_path: str | Path = "./cache/",
    embedders: str = "sbert",
    manager_path: str = "results",
    registry: CorpusEvaluatorRegistry,
) -> None:
    # The corpus part

    sentence = glue_text_field(ds_path)
    # hash the task and models in the embedders, into a unique string name
    embedders_list = embedders.split(",")

    md5_hash = hashlib.md5(
        f"{ds_path}-{ds_split}-{index_name}-{','.join(embedders_list)}".encode("utf-8")
    ).hexdigest()

    LOGGER.info(
        "Unique name for the task and models",
        unique_name=md5_hash,
        ds_path=ds_path,
        ds_split=ds_split,
        index_name=index_name,
        embedders=embedders_list,
    )

    corpus_cache_path = Path(corpus_cache_path)
    corpus_cache_path.mkdir(exist_ok=True, parents=True)
    unique_path = corpus_cache_path / f"{md5_hash}.pkl"

    embedder: Embedder
    if embedders == "sbert":
        embedder = SbertEmbedder(device=device, batch_size=batch_size)
    else:
        embedder = common.ensemble_embedder(
            batch_size=batch_size, embedders=embedders_list
        )
    storage = DatasetsStorage(path=ds_path, split=ds_split)

    corpus: ComposedCorpus
    if unique_path.exists():
        LOGGER.info("Loading corpus from cache", path=unique_path)
        with open(unique_path, "rb") as f:
            corpus = pickle.load(f)
    else:
        corpus = common.composed_corpus(
            batch_size=batch_size,
            device=device,
            index_name=index_name,
            index_threads=index_threads,
            reduced=reduced,
            sentence=sentence,
            embedder=embedder,
            storage=storage,
        )
        with open(unique_path, "wb") as f:
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

    LOGGER.info("Creating adaptor with arguments", sentence=sentence)
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

    optim, optimizer_steps = common.optimizer_and_steps(
        optimizer=optimizer,
        optimizer_steps=optimizer_steps,
        corpus=corpus,
        adaptor=adaptor,
        sobol_steps=sobol_steps,
        device=device,
        task=task,
        acqf=acqf,
        batch_size=batch_size,
        corpus_evals=registry,
    )

    manager = Manager(manager_path)
    manager.run(
        optimizer=optim,
        corpus=corpus,
        model=lm,
        adaptor=adaptor,
        embedder=embedder,
        steps=optimizer_steps,
    )


def glue_text_field(ds_path: str) -> str:
    if "setfit" in ds_path.lower():
        sentence = "text" if ds_path == "SetFit/sst2" else "text1 text2"
    elif ds_path == "SST2":
        sentence = "sentence"
    else:
        raise ValueError(f"Unknown dataset {ds_path}")
    return sentence
