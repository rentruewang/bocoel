import logging
from collections.abc import Sequence
from typing import Literal

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
    ComposedCorpus,
    DatasetsStorage,
    Distance,
    EnsembleEmbedder,
    GlueAdaptor,
    HnswlibIndex,
    HuggingfaceBaseLM,
    HuggingfaceClassifierLM,
    HuggingfaceEmbedder,
    HuggingfaceLogitsLM,
    KMeansOptimizer,
    KMedoidsOptimizer,
    Optimizer,
    Sst2QuestionAnswer,
    Task,
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
    sobol_steps: int = 20,
    index_threads: int = 8,
    optimizer_steps: int = 30,
    device: str = "cpu",
    acqf: str = "ENTROPY",
    task: str = "EXPLORE",
    classification: Literal["logits", "classifier"] = "classifier",
    optimizer: Literal["ax", "kmeans", "kmedoids"] = "ax",
    choices: Sequence[str] = ["negative", "positive"],
) -> None:
    # The corpus part
    LOGGER.info("Loading datasets...", dataset=ds_path, split=ds_split)
    ds = datasets.load_dataset(ds_path)[ds_split]
    storage = DatasetsStorage(ds)

    embedder = ensemble_embedder(batch_size=batch_size)
    idx, sentence, label = idx_sentence_label(ds_path)

    LOGGER.info(
        "Creating corpus with storage and embedder",
        storage=storage,
        embedder=embedder,
        device=device,
    )

    corpus = ComposedCorpus.index_storage(
        storage=storage,
        embedder=embedder,
        keys=sentence.split(),
        index_backend=HnswlibIndex,
        concat=" [SEP] ".join,
        distance=Distance.INNER_PRODUCT,
        threads=index_threads,
    )

    # ------------------------
    # The model part

    lm_cls: type[HuggingfaceBaseLM]
    hf_kwargs = {}
    match classification:
        case "classifier":
            lm_cls = HuggingfaceClassifierLM
            hf_kwargs.update({"choices": choices})
        case "logits":
            lm_cls = HuggingfaceLogitsLM
        case _:
            raise ValueError(f"Unknown classification {classification}")

    LOGGER.info(
        "Creating LM with model",
        lm_cls=lm_cls,
        model=llm_model,
        device=device,
        **hf_kwargs,
    )
    lm = lm_cls(model_path=llm_model, device=device, batch_size=batch_size, **hf_kwargs)

    LOGGER.info(
        "Creating adaptor with arguments", inputs=idx, sentence=sentence, label=label
    )
    adaptor: Adaptor
    if "setfit" in ds_path.lower():
        adaptor = GlueAdaptor(idx, sentence, label)
    elif ds_path == "SST2":
        adaptor = Sst2QuestionAnswer(idx, sentence, label)
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
                lm=lm,
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


def idx_sentence_label(ds_path: str) -> tuple[str, str, str]:
    if "setfit" in ds_path.lower():
        idx = "idx"
        sentence = "text" if ds_path == "SetFit/sst2" else "text1 text2"
        label = "label"
    elif ds_path == "SST2":
        idx = "idx"
        sentence = "sentence"
        label = "label"
    else:
        raise ValueError(f"Unknown dataset {ds_path}")
    return idx, sentence, label


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


if __name__ == "__main__":
    fire.Fire(main)
