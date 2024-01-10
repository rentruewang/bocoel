import datasets
import fire
from ax.modelbridge import Models
from ax.modelbridge.generation_strategy import GenerationStep
from botorch.acquisition import qMaxValueEntropy
from rich import print
from tqdm import tqdm

from bocoel import (
    AxServiceOptimizer,
    BleuScore,
    ComposedCorpus,
    DatasetsStorage,
    Distance,
    HnswlibIndex,
    HuggingfaceLM,
    SBertEmbedder,
    WhiteningIndex,
)


def main(
    *,
    ds_path: str = "lukaemon/bbh",
    ds_name: str,
    ds_key: str = "input",
    ds_target: str = "target",
    sbert_model: str = "all-mpnet-base-v2",
    llm_model: str = "distilgpt2",
    batch_size: int = 16,
    max_len: int = 512,
    device: str = "cpu",
    sobol_steps: int = 20,
    index_threads: int = 8,
    optimizer_steps: int = 30,
    reduced_dim: int = 16,
) -> None:
    # The corpus part
    dataset_dict = datasets.load_dataset(path=ds_path, name=ds_name)
    dataset = dataset_dict["test"]

    dataset_storage = DatasetsStorage(dataset)
    embedder = SBertEmbedder(model_name=sbert_model, device=device)
    corpus = ComposedCorpus.index_storage(
        storage=dataset_storage,
        embedder=embedder,
        key=ds_key,
        klass=WhiteningIndex,
        index_kwargs={
            "distance": Distance.INNER_PRODUCT,
            "remains": reduced_dim,
            "backend": HnswlibIndex,
            "backend_kwargs": {
                "threads": index_threads,
            },
        },
    )

    # ------------------------
    # The model part

    lm = HuggingfaceLM(
        model_path=llm_model, device=device, batch_size=batch_size, max_len=max_len
    )
    score = BleuScore(problem=ds_key, answers=ds_target, lm=lm)

    # ------------------------
    # The optimizer part.
    steps = [
        GenerationStep(Models.SOBOL, num_trials=sobol_steps),
        # GenerationStep(Models.GPMES, num_trials=-1),
        GenerationStep(
            Models.BOTORCH_MODULAR,
            num_trials=-1,
            model_kwargs={
                "torch_device": device,
                "botorch_acqf_class": qMaxValueEntropy,
            },
        ),
    ]

    optim = AxServiceOptimizer.evaluate_corpus(
        corpus=corpus,
        score=score,
        steps=steps,
    )

    for i in tqdm(range(optimizer_steps)):
        state = optim.step()
        print(f"iteration {i}:", state)


if __name__ == "__main__":
    fire.Fire(main)
