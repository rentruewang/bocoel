import fire

from bocoel import Manager, factories


def main() -> None:
    """
    This main function follows the graph here:
    https://rentruewang.github.io/bocoel/references/overview/
    and create components accordingly.
    """

    # First, create the dataset used.
    # 'DATASETS' means that we are going to pull from huggingface's datasets.
    # We are pulling the setfit dataset.
    storage = factories.storage("DATASETS", path="setfit/mnli", split="train")

    # Create an embedder backed by sentence_transformers library.
    embedder = factories.embedder(
        "SBERT", model_name="all-mpnet-base-v2", batch_size=64
    )

    # Using hnswlib because it is a "standalone" index.
    # Other index like `PolarIndex` and `WhiteningIndex` are backed by other indices,
    # which is more complicated.
    corpus = factories.corpus(
        storage=storage,
        embedder=embedder,
        keys=["text1", "text2"],
        index_name="HNSWLIB",
    )

    # MNLI uses the choices "entailment", "neutral", "contradiction".
    MNLI_CHOICES = ["entailment", "neutral", "contradiction"]

    # We are using a sequence classifier from huggingface as the LLM for simplicity.
    llm = factories.classifier(
        "HUGGINGFACE_SEQUENCE",
        model_path="textattack/bert-base-uncased-MNLI",
        batch_size=32,
        choices=MNLI_CHOICES,
    )

    # Create an adaptor for the SST2 dataset.
    # Since we are using MNLI which is part of GLUE.
    # The adaptor is backed by the LLM we just created.
    adaptor = factories.adaptor("GLUE", lm=llm, choices=MNLI_CHOICES)

    # Ax Service Optimizer is the bayesian optimizer that we used.
    optimizer = factories.optimizer("BAYESIAN", corpus=corpus, adaptor=adaptor)

    # Run the optimizer with the manager.
    # Save the results to the "results" folder.
    man = Manager("results")

    # Evaluates the model on the corpus for up to 60 samples.
    man.run(
        steps=60,
        optimizer=optimizer,
        embedder=embedder,
        corpus=corpus,
        model=llm,
        adaptor=adaptor,
    )

    # The end results is saved automatically to the folder "results".
    # Different runs are preserved in the same folder.
    df = Manager.load("results")

    # Analyze df
    ...

    _ = df


if __name__ == "__main__":
    fire.Fire(main)
