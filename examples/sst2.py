import logging
from typing import Literal

import datasets
import fire
import numpy as np
import structlog
from tqdm import tqdm

import bocoel
from bocoel import (
    AcquisitionFunc,
    AxServiceOptimizer,
    ComposedCorpus,
    DatasetsStorage,
    Distance,
    EnsembleEmbedder,
    HnswlibIndex,
    HuggingfaceBaseLM,
    HuggingfaceClassifierLM,
    HuggingfaceEmbedder,
    HuggingfaceLogitsLM,
    Sst2QuestionAnswer,
)

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

LOGGER = structlog.get_logger()


def main(
    *,
    ds_path: str = "SST2",
    ds_split: Literal["train", "validation", "test"] = "train",
    idx: str = "idx",
    sentence: str = "sentence",
    label: str = "label",
    sbert_model: str = "all-mpnet-base-v2",
    llm_model: str = "textattack/roberta-base-SST-2",
    batch_size: int = 16,
    device: str = "cpu",
    sobol_steps: int = 20,
    index_threads: int = 8,
    optimizer_steps: int = 30,
    acqf: str = "ENTROPY",
    classification: Literal["logits", "classifier"] = "classifier",
) -> None:
    # The corpus part
    LOGGER.info("Loading datasets...", dataset=ds_path, split=ds_split)

    ds = datasets.load_dataset(ds_path)[ds_split]
    storage = DatasetsStorage(ds)

    LOGGER.info(
        "Creating embedder",
        model=sbert_model,
        device=device,
    )
    # embedder = SbertEmbedder(
    #     model_name=sbert_model, device=device, batch_size=batch_size
    # )

    embedders = []
    for model in [  # "textattack/bert-base-uncased-SST-2",
        #   "textattack/roberta-base-SST-2",
        #   "textattack/albert-base-v2-SST-2",
        #   "textattack/xlnet-large-cased-SST-2",
        #   "textattack/xlnet-base-cased-SST-2",
        #   "textattack/facebook-bart-large-SST-2",
        #   "textattack/distilbert-base-uncased-SST-2",
        "textattack/distilbert-base-cased-SST-2"
    ]:
        embedders.append(
            HuggingfaceEmbedder(path=model, device=device, batch_size=batch_size)
        )
    embedder = EnsembleEmbedder(embedders)

    LOGGER.info(
        "Creating corpus with storage and embedder",
        storage=storage,
        embedder=embedder,
        device=device,
    )
    corpus = ComposedCorpus.index_storage(
        storage=storage,
        embedder=embedder,
        key=sentence,
        index_backend=HnswlibIndex,
        distance=Distance.INNER_PRODUCT,
        # whitening_backend=HnswlibIndex,
        threads=index_threads,
    )

    # ------------------------
    # The model part

    lm_cls: HuggingfaceBaseLM
    if classification == "classifier":
        lm_cls = HuggingfaceClassifierLM
    elif classification == "logits":
        lm_cls = HuggingfaceLogitsLM
    else:
        raise ValueError
    LOGGER.info("Creating LM with model", model=llm_model, device=device)
    lm = lm_cls(model_path=llm_model, device=device, batch_size=batch_size)
    # LOGGER.info("Creating LM with model", model=llm_model, device=device)
    # lm = HuggingfaceClassifierLM(
    #     model_path=llm_model, device=device, batch_size=batch_size
    # )

    LOGGER.info(
        "Creating adaptor with arguments", inputs=idx, sentence=sentence, label=label
    )
    adaptor = Sst2QuestionAnswer(idx=idx, sentence=sentence, label=label)

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
    optim = bocoel.evaluate_corpus(
        AxServiceOptimizer,
        corpus=corpus,
        lm=lm,
        adaptor=adaptor,
        sobol_steps=sobol_steps,
        device=device,
        acqf=AcquisitionFunc.lookup(acqf),
    )

    scores = []
    for i in tqdm(range(optimizer_steps)):
        state = optim.step()
        LOGGER.info("iteration {i}: {state}", i=i, state=state)
        scores.append(state[i])

    print(np.average(scores))


if __name__ == "__main__":
    fire.Fire(main)
