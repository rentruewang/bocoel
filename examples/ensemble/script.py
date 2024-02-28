import itertools

from torch import cuda

from . import glue

DS_PATHS = ["SetFit/mrpc"]
DS_SPLITS = ["validation"]
BATCH_SIZE = 1
OPTIMIZERS = [
    {"name": "brute", "steps": 408},
    {"name": "ax", "steps": 408},
    *[{"name": "kmeans", "steps": i} for i in range(8, 409, 10)],
    *[{"name": "kmedoids", "steps": i} for i in range(8, 409, 10)],
    {"name": "random", "steps": 408},
]
MODELS = [
    "gpt2",
    "EleutherAI/pythia-70m",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "princeton-nlp/Sheared-LLaMA-1.3B",
    "princeton-nlp/Sheared-LLaMA-2.7B",
]
CLASSIFICATION = ["logits"]
INDICES = ["polar", "whitening", "inverse_cdf"]
EMBEDDERS = [
    "sbert",
    # "textattack/bert-base-uncased-MRPC,textattack/roberta-base-MRPC,textattack/distilbert-base-uncased-MRPC,textattack/albert-base-v2-MRPC,textattack/distilbert-base-cased-MRPC,textattack/xlnet-large-cased-MRPC,textattack/xlnet-base-cased-MRPC"
]
INDEX_THREADS = 8
REDUCED = 32
DEVICE = "cuda" if cuda.is_available() else "cpu"


def main():
    for (
        ds_path,
        ds_split,
        classification,
        embedder,
        llm_model,
        optim,
        index_name,
    ) in itertools.product(
        DS_PATHS, DS_SPLITS, CLASSIFICATION, EMBEDDERS, MODELS, OPTIMIZERS, INDICES
    ):
        glue.main(
            ds_path=ds_path,
            ds_split=ds_split,
            llm_model=llm_model,
            optimizer=optim["name"],
            optimizer_steps=optim["steps"],
            batch_size=BATCH_SIZE,
            index_name=index_name,
            classification=classification,
            index_threads=INDEX_THREADS,
            embedders=embedder,
            reduced=REDUCED,
            device=DEVICE,
        )


if __name__ == "__main__":
    for _ in range(10):
        main()
